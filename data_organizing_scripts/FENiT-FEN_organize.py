"""
FENiT-FEN_organize.py
================================
Processes the FENiT-FEN YOLO-pose dataset (images + .txt labels) into two outputs:

  OUTPUT 1 — Board corners for corner detection training:
    organized_data/corner_detection/images/train/   ← board images
    organized_data/corner_detection/labels/train/   ← class 0 only YOLO pose labels

  OUTPUT 2 — Piece crops for classification training:
    organized_data/classification/physical/
        wK/ wQ/ wR/ wB/ wN/ wP/
        bK/ bQ/ bR/ bB/ bN/ bP/

Preprocessing on piece crops (identical to reference script):
    - Tight bbox crop from YOLO cx/cy/w/h
    - CLAHE on V channel (HSV) for dark piece visibility
    - Convert to grayscale
    - Resize to 64x64 (LANCZOS)
    - Augmentation: brightness, contrast, rotation, flip, blur, noise
    - wN and bB get 3x extra augmentation (rare classes)

Input structure (your dataset):
    <raw_dir>/
        images/   ← .jpg files
        labels/   ← .txt files (same stem as image)

Balancing strategy:
    - A TARGET_COUNT is computed as the median raw count across all classes.
    - Per-class aug multiplier = ceil(TARGET_COUNT / raw_count) - 1  (min 0).
    - wN and bB get RARE_BONUS extra augment copies on top (tougher classification).
    - wP and bP are naturally capped — if raw count >= TARGET_COUNT, aug = 0.
    - Final counts land within ~±20% of TARGET_COUNT across all classes.

Usage:
    python FENiT-FEN_organize.py
    python FENiT-FEN_organize.py --raw ./raw_data/FENiT-FEN --seed 42
"""

import os
import re
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()/ ".."

# Default raw dataset location 
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw_data" / "FENiT-FEN"

# Corner detection output — merges INTO existing corner_detection folder
CORNER_OUT     = PROJECT_ROOT / "organized_data" / "corner_detection"

# Piece classification output
PIECE_OUT      = PROJECT_ROOT / "organized_data" / "classification" / "physical"

# Preprocessing
CROP_SIZE      = 64       # output piece crop size (px)
MIN_CROP_PX    = 20       # skip crops smaller than this

# ── Balancing strategy ────────────────────────────────────────────────────
# TARGET_COUNT: desired total images per class after augmentation.
# Per-class aug multiplier = max(0, ceil(TARGET / raw_count) - 1)
# All classes land close to TARGET → total dataset ~TARGET*12 images.
# Default TARGET=2700 → ~32400 total crops (well within 30-35k goal).
# Adjust TARGET via --target CLI arg.
TARGET_COUNT   = 2700

# Fallback: pass --augment N on CLI to use a flat multiplier for all classes
# (overrides dynamic balancing — useful for ablation tests)
AUGMENT_N      = None     # None = dynamic balancing (recommended)

# YOLO class id → our label
# Based on data.yaml from the dataset
YOLO_TO_LABEL = {
    0:  "Board",   # handled separately → corner detection
    1:  "wK",
    2:  "wP",
    3:  "bP",
    4:  "bK",
    5:  "wQ",
    6:  "wB",
    7:  "wN",
    8:  "wR",
    9:  "bB",
    10: "bR",
    11: "bN",
    12: "bQ",
}

PIECE_CLASSES = [
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]


# ─────────────────────────────────────────────
# LABEL PARSING
# ─────────────────────────────────────────────

def parse_yolo_pose_label(txt_path):
    """
    Parse a YOLO pose label file.
    Each object = 17 tokens:
        class cx cy w h  kx1 ky1 v1  kx2 ky2 v2  kx3 ky3 v3  kx4 ky4 v4

    Returns list of dicts:
        {cls, cx, cy, bw, bh, kpts: [(kx,ky,vis), ...]}
    """
    content = txt_path.read_text().strip()
    if not content:
        return []

    tokens = list(map(float, content.split()))

    # Validate token count is multiple of 17
    if len(tokens) % 17 != 0:
        print(f"  [WARN] Unexpected token count ({len(tokens)}) in {txt_path.name} — skipping")
        return []

    objects = []
    for i in range(0, len(tokens), 17):
        cls  = int(tokens[i])
        cx   = tokens[i+1]
        cy   = tokens[i+2]
        bw   = tokens[i+3]
        bh   = tokens[i+4]
        kpts = []
        for k in range(4):
            kx  = tokens[i+5 + k*3]
            ky  = tokens[i+6 + k*3]
            vis = tokens[i+7 + k*3]
            kpts.append((kx, ky, vis))
        objects.append({
            'cls' : cls,
            'cx'  : cx, 'cy': cy,
            'bw'  : bw, 'bh': bh,
            'kpts': kpts,
        })

    return objects


def build_corner_label_line(obj):
    """
    Rebuild a YOLO pose label line for class 0 (board) only.
    Output: '0 cx cy bw bh kx1 ky1 v1 kx2 ky2 v2 kx3 ky3 v3 kx4 ky4 v4'
    """
    parts = [
        '0',
        f"{obj['cx']:.10f}", f"{obj['cy']:.10f}",
        f"{obj['bw']:.10f}", f"{obj['bh']:.10f}",
    ]
    for kx, ky, vis in obj['kpts']:
        parts += [f"{kx:.10f}", f"{ky:.10f}", f"{vis:.1f}"]
    return ' '.join(parts)


def yolo_to_pixel_bbox(cx, cy, bw, bh, W, H):
    """
    Convert normalized YOLO bbox to pixel coordinates.
    Returns (xmin, ymin, xmax, ymax) clamped to image bounds.
    """
    xmin = int((cx - bw / 2) * W)
    ymin = int((cy - bh / 2) * H)
    xmax = int((cx + bw / 2) * W)
    ymax = int((cy + bh / 2) * H)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W, xmax)
    ymax = min(H, ymax)
    return xmin, ymin, xmax, ymax


# ─────────────────────────────────────────────
# PREPROCESSING  (identical to reference script)
# ─────────────────────────────────────────────

def apply_clahe(crop_img):
    """
    CLAHE on V channel of HSV — boosts dark piece visibility.
    Then convert to grayscale to match synthetic pipeline.
    """
    img_np  = np.array(crop_img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_rgb).convert("L")   # grayscale


def crop_and_preprocess(img, xmin, ymin, xmax, ymax):
    """
    Crop piece from image, apply CLAHE, resize to 64x64.
    Returns None if crop is too small.
    """
    w = xmax - xmin
    h = ymax - ymin

    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return None

    crop = img.crop((xmin, ymin, xmax, ymax))
    crop = apply_clahe(crop)
    crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    return crop


# ─────────────────────────────────────────────
# AUGMENTATION  (identical to reference script)
# ─────────────────────────────────────────────

def augment_crop(crop_img):
    """
    Random augmentation for a 64x64 grayscale crop.
    Matches reference script augmentation suite exactly.
    """
    img = crop_img.copy()

    # Brightness
    if random.random() > 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))

    # Contrast
    if random.random() > 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    # Rotation ±12°
    if random.random() > 0.4:
        img = img.rotate(random.uniform(-12, 12), fillcolor=128)

    # Horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Gaussian blur
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    # Noise
    if random.random() > 0.5:
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    return img


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def save_piece_crop(crop, label, crop_idx, aug_idx=None):
    """Save a piece crop to the correct class subfolder."""
    cls_dir = PIECE_OUT / label
    cls_dir.mkdir(parents=True, exist_ok=True)
    if aug_idx is None:
        fname = f"pose_{crop_idx:06d}_base.png"
    else:
        fname = f"pose_{crop_idx:06d}_aug{aug_idx}.png"
    crop.save(cls_dir / fname)


def save_corner_files(img_path, corner_label_line, file_idx):
    """
    Copy image and write corner-only label into corner_detection/images/train
    and corner_detection/labels/train.
    Uses a unique prefix to avoid filename collisions with kaggle dataset.
    """
    img_dst_dir = CORNER_OUT / "images" / "train"
    lbl_dst_dir = CORNER_OUT / "labels" / "train"
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    # Prefix with 'pose_' to avoid collision with G000_IMG000 naming
    new_stem = f"pose_{file_idx:06d}"
    shutil.copy2(img_path, img_dst_dir / f"{new_stem}.jpg")
    (lbl_dst_dir / f"{new_stem}.txt").write_text(corner_label_line)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def compute_aug_map(raw_dir, img_files, labels_dir, flat_augment_n):
    """
    PASS 1 — count raw base crops per class without saving anything.
    Returns a dict {label: n_aug_copies} using dynamic balancing,
    or flat_augment_n for every class if flat_augment_n is not None.
    """
    import math

    raw_counts = defaultdict(int)
    for img_path in img_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        objects = parse_yolo_pose_label(lbl_path)
        for obj in objects:
            label = YOLO_TO_LABEL.get(obj['cls'])
            if label and label != "Board":
                raw_counts[label] += 1

    if flat_augment_n is not None:
        # CLI override: flat multiplier for all classes
        return {cls: flat_augment_n for cls in PIECE_CLASSES}, raw_counts

    # Dynamic: use TARGET_COUNT to compute per-class aug multiplier
    # n_aug = ceil(TARGET / raw) - 1, so total ≈ TARGET per class
    target = TARGET_COUNT
    aug_map = {}
    for cls in PIECE_CLASSES:
        raw = raw_counts.get(cls, 0)
        if raw == 0:
            aug_map[cls] = 0
            continue
        aug_map[cls] = max(0, math.ceil(target / raw) - 1)

    return aug_map, raw_counts


def organize(raw_dir, augment_n, seed):
    random.seed(seed)
    np.random.seed(seed)

    raw_path   = Path(raw_dir)
    images_dir = raw_path / "images"
    labels_dir = raw_path / "labels"

    # ── Validate source ───────────────────────
    assert images_dir.exists(), f"[ERROR] images/ not found at {images_dir}"
    assert labels_dir.exists(), f"[ERROR] labels/ not found at {labels_dir}"

    img_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG"))
    assert img_files, f"[ERROR] No .jpg images found in {images_dir}"

    # ── PASS 1: compute per-class aug targets ─
    print("  [Pass 1] Counting raw base crops for balancing...")
    aug_map, raw_counts = compute_aug_map(raw_path, img_files, labels_dir, augment_n)

    # ── Header ────────────────────────────────
    mode_str = f"flat={augment_n}" if augment_n is not None else f"dynamic (target={TARGET_COUNT}/class)"
    print("=" * 58)
    print("  FENiT-FEN Dataset Organizer")
    print("=" * 58)
    print(f"  Source          : {raw_path}")
    print(f"  Images found    : {len(img_files)}")
    print(f"  Corner out      : {CORNER_OUT}")
    print(f"  Piece crops out : {PIECE_OUT}")
    print(f"  Aug mode        : {mode_str}")
    print(f"  Per-class aug targets:")
    for cls in PIECE_CLASSES:
        raw = raw_counts.get(cls, 0)
        n   = aug_map.get(cls, 0)
        est = raw * (1 + n)
        print(f"    {cls:5s}: raw={raw:5d}  aug_copies={n:2d}  est_total≈{est:6d}")
    print("=" * 58)

    # ── Create output dirs ────────────────────
    for cls in PIECE_CLASSES:
        (PIECE_OUT / cls).mkdir(parents=True, exist_ok=True)

    # ── Stats ─────────────────────────────────
    piece_counts   = defaultdict(int)
    corner_saved   = 0
    piece_saved    = 0
    skipped_crops  = 0
    skipped_no_lbl = 0
    skipped_no_brd = 0
    crop_idx       = 0
    file_idx       = 0

    # ── PASS 2: process and save ──────────────
    for img_path in img_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"

        if not lbl_path.exists():
            print(f"  [WARN] No label for {img_path.name} — skipping")
            skipped_no_lbl += 1
            continue

        objects = parse_yolo_pose_label(lbl_path)
        if not objects:
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path.name}: {e}")
            continue

        W, H = pil_img.size

        # ── OUTPUT 1: Board corners ───────────
        board_objs = [o for o in objects if o['cls'] == 0]
        if not board_objs:
            skipped_no_brd += 1
        else:
            board_obj         = board_objs[0]
            corner_label_line = build_corner_label_line(board_obj)
            save_corner_files(img_path, corner_label_line, file_idx)
            corner_saved += 1
            file_idx     += 1

        # ── OUTPUT 2: Piece crops ─────────────
        piece_objs = [o for o in objects if o['cls'] != 0]

        for obj in piece_objs:
            label = YOLO_TO_LABEL.get(obj['cls'])
            if label is None or label == "Board":
                continue

            xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(
                obj['cx'], obj['cy'], obj['bw'], obj['bh'], W, H
            )

            crop = crop_and_preprocess(pil_img, xmin, ymin, xmax, ymax)
            if crop is None:
                skipped_crops += 1
                continue

            # Save base crop
            save_piece_crop(crop, label, crop_idx)
            piece_counts[label] += 1
            piece_saved += 1

            # Per-class augmentation (dynamic or flat)
            n_aug = aug_map.get(label, 0)
            for aug_i in range(n_aug):
                aug = augment_crop(crop)
                save_piece_crop(aug, label, crop_idx, aug_i)
                piece_counts[label] += 1
                piece_saved += 1

            crop_idx += 1

    # ── Summary ───────────────────────────────
    print(f"\n  ── Results ─────────────────────────────")
    print(f"  Images processed     : {len(img_files) - skipped_no_lbl}")
    print(f"  Skipped (no label)   : {skipped_no_lbl}")
    print(f"  Skipped (no board)   : {skipped_no_brd}")
    print(f"  Corner labels saved  : {corner_saved}  → {CORNER_OUT / 'images' / 'train'}")
    print(f"  Piece crops saved    : {piece_saved}   → {PIECE_OUT}")
    print(f"  Piece crops skipped  : {skipped_crops} (bbox < {MIN_CROP_PX}px)")

    print(f"\n  ── Per-class piece counts ───────────────")
    max_count = max(piece_counts.values()) if piece_counts else 1
    for cls in PIECE_CLASSES:
        count = piece_counts.get(cls, 0)
        bar   = "█" * int(30 * count / max(max_count, 1))
        flag  = "  ← RARE" if 0 < count < 50 else ""
        print(f"    {cls:5s}: {count:5d}  {bar}{flag}")

    if piece_counts:
        counts    = list(piece_counts.values())
        min_c     = min(counts)
        max_c     = max(counts)
        ratio     = max_c / max(min_c, 1)
        print(f"\n  Class imbalance ratio: {ratio:.1f}x  (min:{min_c}  max:{max_c})")
        if ratio > 8:
            print("  [WARN] High imbalance — check augmentation settings")

    print("\n" + "=" * 58)
    print("  Done!")
    print(f"  Corner data → organized_data/corner_detection/")
    print(f"  Piece crops → organized_data/classification/physical/")
    print("=" * 58)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Organize FENiT-FEN YOLO-pose dataset into corner detection + piece crops."
    )
    p.add_argument("--raw",     default=str(DEFAULT_RAW_DIR),
                   help="Raw dataset root containing images/ and labels/")
    p.add_argument("--augment", type=int, default=None,
                   help="Flat augmented copies per crop (default: dynamic balancing)")
    p.add_argument("--target",  type=int, default=TARGET_COUNT,
                   help=f"Target total images per class for dynamic balancing (default {TARGET_COUNT})")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed (default 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Apply --target override if provided
    import sys
    if "--target" in sys.argv:
        TARGET_COUNT = args.target  # noqa: F811

    organize(
        raw_dir   = args.raw,
        augment_n = args.augment,
        seed      = args.seed,
    )
