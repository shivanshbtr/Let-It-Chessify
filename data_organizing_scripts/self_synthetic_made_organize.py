"""
Chess OCR — Synthetic Square Dataset Builder
=================================================
Runs the NEW 5-tier robust digital-board detection pipeline --
find_board_grid_robust() (chessboard_grid_segmentation.py) -> per-cell
perspective slice_squares() -> preprocess_synthetic_crop ->
resize_to_model_input -- over every image in the raw synthetic dataset,
reads the matching FEN label file, and sorts each of the 64 resulting
square crops into a per-class folder ready for classifier training.

Why this matters for a DATASET-BUILDING script specifically: whatever
crops this script produces are what the classifier is trained on, and
whatever crops main.py's /detect endpoint produces at inference time are
what it's evaluated on. If those two pipelines diverge -- different
detector, different slicing, different preprocessing order -- the model
is trained on subtly different pixel data than it sees live, which is a
silent, hard-to-diagnose accuracy bug. This script is written to mirror
main.py's actual detect_and_slice_digital_board() / slice_digital_board_from_grid()
exactly (same detector, same cell_px, same preprocessing call order), not
just "a reasonable equivalent" -- see SLICE_CELL_PX below in particular.

This script does NOT modify preprocessing.py / warp.py /
chessboard_grid_segmentation.py in any way -- it only imports and calls
them, so whatever is currently sitting in detection/preprocessing.py is
exactly what gets applied.

Directory layout expected:

    <cur_dir>/build_synthetic_squares_dataset.py    <- this script
    <cur_dir>/../backend/detection/...      <- the backend project (unmodified)
    <cur_dir>/../raw_data/self_synthetic_made/images/<name>.<ext>
    <cur_dir>/../raw_data/self_synthetic_made/labels/<name>.txt   (contains a FEN string)

Output:

    <this_dir>/../organized_data/classification/synthetic/<class>/scr-<name>-<square>.png

    <class> in: bB bK bN bP bQ bR wB wK wN wP wQ wR empty

Usage:
    python build_synthetic_squares_dataset.py
    python build_synthetic_squares_dataset.py --images-dir ... --labels-dir ... --out-dir ...
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import cv2
import chess
from PIL import Image, ImageEnhance

# ── Locate + import the backend detection package ──────────────────────────────
# Layout: this script sits in <cur_dir>/, and the backend project root
# (the folder directly containing the 'backend' package, i.e. the folder
# that has backend/detection/*.py inside it) sits at <cur_dir>/../backend
THIS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = THIS_DIR.parent / ""
sys.path.insert(0, str(BACKEND_ROOT))

from backend.detection.chessboard_grid_segmentation import (
    find_board_grid_robust,
    slice_squares as slice_squares_from_grid,
)
from backend.detection.preprocessing import (
    preprocess_synthetic_crop,
    resize_to_model_input,
)

# ── FEN piece -> output class folder name ──────────────────────────────────────

_PIECE_TO_CLASS = {
    (chess.WHITE, chess.PAWN):   "wP",
    (chess.WHITE, chess.KNIGHT): "wN",
    (chess.WHITE, chess.BISHOP): "wB",
    (chess.WHITE, chess.ROOK):   "wR",
    (chess.WHITE, chess.QUEEN):  "wQ",
    (chess.WHITE, chess.KING):   "wK",
    (chess.BLACK, chess.PAWN):   "bP",
    (chess.BLACK, chess.KNIGHT): "bN",
    (chess.BLACK, chess.BISHOP): "bB",
    (chess.BLACK, chess.ROOK):   "bR",
    (chess.BLACK, chess.QUEEN):  "bQ",
    (chess.BLACK, chess.KING):   "bK",
}

ALL_CLASSES = sorted(_PIECE_TO_CLASS.values()) + ["empty"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

NUM_AUGMENTS_PER_SQUARE = 10   # extra augmented variants generated per square, in ADDITION to the original
NUM_AUGMENTS_EMPTY      = 0    # 'empty' class gets NO augments — it's already the majority
                                # class by a wide margin (most squares on a board are empty),
                                # so augmenting it further would only widen the class imbalance.

# Must match main.py's slice_digital_board_from_grid() exactly -- see this
# script's module docstring for why train/inference crop-generation
# divergence is a real (and silent) accuracy risk. 96px preserves more raw
# detail from the source image than the final 64x64 model input before the
# LANCZOS downsize in resize_to_model_input.
SLICE_CELL_PX = 96


def augment_crop(img: Image.Image, seed: int) -> Image.Image:
    """
    Light, label-preserving augmentation applied to an already-preprocessed
    64x64 'L' (grayscale) square crop. Does NOT touch warp/preprocessing.py —
    this runs strictly after the pipeline's output, as an extra
    dataset-diversity step for retraining on multiple board backgrounds.

    Mix: small rotation, brightness jitter, contrast jitter, mild gaussian
    noise. Kept intentionally gentle so piece silhouettes remain intact and
    the augmented crop stays consistent with what preprocessing.py produces.
    """
    rng = random.Random(seed)

    # small rotation (+/- 6 degrees). Rotating in-place with a solid fillcolor
    # exposes a flat gray wedge at the corner (visible artifact, especially
    # bad on 'empty' squares where it can look like a hard edge). Fix: pad by
    # reflecting the image outward first, rotate, then crop back to the
    # original size — no fill color ever becomes visible.
    angle = rng.uniform(-6, 6)
    w, h = img.size
    pad = max(w, h) // 2
    padded = Image.fromarray(
        np.pad(np.array(img), pad, mode="reflect")
    )
    rotated = padded.rotate(angle, resample=Image.BILINEAR)
    out = rotated.crop((pad, pad, pad + w, pad + h))

    # brightness jitter
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))

    # contrast jitter
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))

    # mild gaussian noise
    arr = np.array(out, dtype=np.float32)
    noise = np.random.RandomState(seed).normal(0, 6.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr, mode="L")

    return out


# ── Board detection + slicing (mirrors main.py's detect_and_slice_digital_board) ──

def detect_grid(img_np):
    """
    Runs the 5-tier pipeline, identical call to main.py's
    detect_and_slice_digital_board(). Returns (grid, tier_name, score) or
    (None, None, None) if all 5 tiers failed.

    return_info=True costs nothing extra here (same internal work either
    way) and is valuable for dataset QA -- if a systematic slice of this
    synthetic dataset only resolves via Tier 4/5 (no perspective
    correction, weaker self-consistency signal than Tiers 1-3), that's
    worth knowing before trusting those crops as training labels, rather
    than silently accepting whatever came out.
    """
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    try:
        grid, tier, score = find_board_grid_robust(img_bgr, return_info=True)
        return grid, tier, score
    except RuntimeError:
        return None, None, None


def slice_digital_board(img_np, grid):
    """
    Slices all 64 cells via the pipeline's own per-cell perspective warp
    (slice_squares_from_grid), then applies the EXACT SAME preprocessing
    recipe main.py's slice_digital_board_from_grid() applies at inference
    time: preprocess_synthetic_crop + resize_to_model_input, in that
    order. Do not reorder or substitute steps here without making the
    identical change in main.py -- see module docstring.

    Returns: dict {square_index (python-chess convention) -> PIL Image, mode "L", 64x64}
    """
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cells = slice_squares_from_grid(img_bgr, grid, cell_px=SLICE_CELL_PX)  # {(row,col): BGR np array}

    crops = {}
    for (row, col), cell_bgr in cells.items():
        rank     = 7 - row   # row 0 = rank 8 (top) -- matches python-chess / main.py convention
        file_idx = col
        sq_idx   = chess.square(file_idx, rank)

        cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(cell_rgb)

        processed = preprocess_synthetic_crop(pil_crop, file_idx, rank)
        processed = resize_to_model_input(processed, size=64)
        crops[sq_idx] = processed

    return crops


# ── FEN parsing ──────────────────────────────────────────────────────────────

def load_fen(label_path: Path) -> str:
    text = label_path.read_text().strip()
    if not text:
        raise ValueError(f"Empty label file: {label_path}")
    # Label file may contain a full FEN ("... w KQkq - 0 1") or just the
    # piece-placement field. Either way we only need the placement field.
    placement = text.split()[0]
    return placement


def class_for_square(base_board: chess.BaseBoard, sq_idx: int) -> str:
    piece = base_board.piece_at(sq_idx)
    if piece is None:
        return "empty"
    return _PIECE_TO_CLASS[(piece.color, piece.piece_type)]


# ── Main per-image processing ──────────────────────────────────────────────────

def process_image(image_path: Path, label_path: Path, out_dir: Path, stats: dict) -> None:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    grid, tier, score = detect_grid(img_np)
    if grid is None:
        print(f"  [SKIP] {image_path.name}: no board grid detected (all 5 tiers failed)")
        stats["skipped_no_grid"] += 1
        return

    crops = slice_digital_board(img_np, grid)

    fen_placement = load_fen(label_path)
    base_board = chess.BaseBoard(fen_placement)

    stem = image_path.stem
    for sq_idx, crop in crops.items():
        cls = class_for_square(base_board, sq_idx)
        sq_name = chess.square_name(sq_idx)

        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        # original (unaugmented) crop, straight from the pipeline
        out_name = f"scr-{stem}-{sq_name}.png"
        crop.save(cls_dir / out_name)
        stats["squares_written"] += 1
        stats["per_class"][cls] = stats["per_class"].get(cls, 0) + 1

        # augmented variants — 'empty' squares get fewer augments than piece squares
        num_augments = NUM_AUGMENTS_EMPTY if cls == "empty" else NUM_AUGMENTS_PER_SQUARE
        for aug_i in range(1, num_augments + 1):
            seed = hash((stem, sq_name, aug_i)) & 0xFFFFFFFF
            aug_crop = augment_crop(crop, seed)
            aug_name = f"scr-{stem}-{sq_name}-aug{aug_i}.png"
            aug_crop.save(cls_dir / aug_name)
            stats["squares_written"] += 1
            stats["augmented_written"] += 1
            stats["per_class"][cls] = stats["per_class"].get(cls, 0) + 1

    stats["images_processed"] += 1
    stats["per_tier"][tier] = stats["per_tier"].get(tier, 0) + 1
    print(f"  [OK]   {image_path.name}  (tier={tier}, score={score:.2f})")


def find_label_for_image(image_path: Path, labels_dir: Path) -> Path | None:
    candidate = labels_dir / f"{image_path.stem}.txt"
    return candidate if candidate.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path,
        default=THIS_DIR.parent / "raw_data" / "self_synthetic_made" / "images",
    )
    parser.add_argument(
        "--labels-dir", type=Path,
        default=THIS_DIR.parent / "raw_data" / "self_synthetic_made" / "labels",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=THIS_DIR.parent / "organized_data" / "classification" / "synthetic",
    )
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir
    out_dir    = args.out_dir

    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"ERROR: labels dir not found: {labels_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in ALL_CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"ERROR: no images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {images_dir}")
    print(f"Writing squares to {out_dir}\n")

    stats = {
        "images_processed": 0,
        "skipped_no_grid": 0,
        "skipped_no_label": 0,
        "skipped_error": 0,
        "squares_written": 0,
        "augmented_written": 0,
        "per_class": {},
        "per_tier": {},
    }

    for image_path in image_paths:
        label_path = find_label_for_image(image_path, labels_dir)
        if label_path is None:
            print(f"  [SKIP] {image_path.name}: no matching label file "
                  f"({labels_dir / (image_path.stem + '.txt')} not found)")
            stats["skipped_no_label"] += 1
            continue

        try:
            process_image(image_path, label_path, out_dir, stats)
        except Exception as e:
            print(f"  [ERROR] {image_path.name}: {e}")
            traceback.print_exc()
            stats["skipped_error"] += 1

    print("\n── Summary ──────────────────────────────────────────")
    print(f"Images processed:      {stats['images_processed']}")
    print(f"Skipped (no grid):     {stats['skipped_no_grid']}")
    print(f"Skipped (no label):    {stats['skipped_no_label']}")
    print(f"Skipped (error):       {stats['skipped_error']}")
    print(f"Augmentations/square:  {NUM_AUGMENTS_PER_SQUARE} (piece classes), {NUM_AUGMENTS_EMPTY} (empty class)")
    original_count = stats["squares_written"] - stats["augmented_written"]
    print(f"Original squares:      {original_count}")
    print(f"Augmented squares:     {stats['augmented_written']}")
    print(f"TOTAL images written:  {stats['squares_written']}")
    print("\nPer-class counts (original + augmented):")
    for cls in ALL_CLASSES:
        print(f"  {cls:6s} {stats['per_class'].get(cls, 0)}")
    print("\nPer-tier resolution counts (QA signal -- a large share of Tier")
    print("4/5 hits means many images had no perspective correction applied;")
    print("worth spot-checking those specific images' crops before trusting")
    print("them as training labels):")
    for tier_name, count in sorted(stats["per_tier"].items(), key=lambda kv: kv[0]):
        print(f"  {tier_name:24s} {count}")


if __name__ == "__main__":
    main()
