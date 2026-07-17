#!/usr/bin/env python3
"""
organize_piece_detection_data.py

Merges two raw chess datasets into a single-class ("piece") object-detection
dataset in standard YOLO format.

Source 1: raw_data/Chess Piece Detection/{images,annotations}
    - Pascal VOC XML annotations, one <object> per piece, class names like
      "white-knight", "black-queen", etc. Pixel-space bndbox coords.

Source 2: raw_data/FENiT-FEN/{images,labels}
    - YOLO-pose style .txt labels: `class cx cy w h kp1x kp1y kp1v ... kp4v`
      (normalized 0-1 coords). One class is the *board* (a single huge box
      per image) which must be excluded. Everything else is a piece.

Output: organized_data/piece_detection/
    images/train, images/val
    labels/train, labels/val
    data.yaml

Usage:
    python organize_piece_detection_data.py
    python organize_piece_detection_data.py --val-split 0.15 --seed 42
    python organize_piece_detection_data.py --dry-run

By default all paths are resolved relative to the project root, which is
assumed to be the parent of this script's parent directory
(main_project_dir/data_organizing_scripts/this_script.py). Override with
--project-root if your layout differs.
"""

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # only needed as a fallback if VOC xml lacks <size>

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
PIECE_CLASS_ID = 0  # single unified class id in the output dataset
PIECE_CLASS_NAME = "piece"

# Known FENiT-FEN class map (confirmed against real label files). Used as a
# fallback whenever no classes.txt/notes.json/data.yaml is found alongside
# the labels directory. Class 0 ("Board") is excluded; everything else 1-12
# is some piece and gets collapsed to PIECE_CLASS_ID.
DEFAULT_FEN_CLASS_MAP = {
    0: "Board",
    1: "wK",
    2: "wP",
    3: "bP",
    4: "bK",
    5: "wQ",
    6: "wB",
    7: "wN",
    8: "wR",
    9: "bB",
    10: "bR",
    11: "bN",
    12: "bQ",
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    """Case-insensitive match of `stem` against files in images_dir."""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # fall back to a slower case-insensitive scan (handles odd casing)
    lower_target = stem.lower()
    for f in images_dir.iterdir():
        if f.is_file() and f.stem.lower() == lower_target and f.suffix.lower() in {e.lower() for e in IMAGE_EXTS}:
            return f
    return None


def get_image_size(img_path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError(
            f"Pillow is required to read image dimensions for {img_path} "
            f"(install with `pip install pillow --break-system-packages`)"
        )
    with Image.open(img_path) as im:
        return im.width, im.height


# --------------------------------------------------------------------------- #
# Source 1: Pascal VOC XML  ->  YOLO single-class lines
# --------------------------------------------------------------------------- #

def collect_voc_pairs(images_dir: Path, annots_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for xml_path in sorted(annots_dir.glob("*.xml")):
        img_path = find_image_for_stem(images_dir, xml_path.stem)
        if img_path is None:
            print(f"  [voc] WARNING: no image found for annotation {xml_path.name}, skipping")
            continue
        pairs.append((img_path, xml_path))
    return pairs


def voc_xml_to_yolo_lines(xml_path: Path, img_path: Path) -> tuple[list[str], int]:
    """Returns (yolo_lines, num_boxes_dropped_as_board)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    if size_el is not None and size_el.find("width") is not None and size_el.find("height") is not None:
        img_w = int(size_el.find("width").text)
        img_h = int(size_el.find("height").text)
    else:
        img_w, img_h = get_image_size(img_path)

    if img_w <= 0 or img_h <= 0:
        img_w, img_h = get_image_size(img_path)

    lines = []
    dropped_board = 0
    for obj in root.findall("object"):
        name_el = obj.find("name")
        name = (name_el.text or "").strip().lower() if name_el is not None else ""
        if name == "board":
            dropped_board += 1
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # clip to image bounds defensively
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{PIECE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines, dropped_board


# --------------------------------------------------------------------------- #
# Source 2: YOLO-pose txt  ->  YOLO single-class lines
# --------------------------------------------------------------------------- #

def load_class_names(labels_dir: Path) -> dict[int, str] | None:
    """
    Look for a class-name map near the labels directory
    (classes.txt, notes.json from labelImg/roboflow, or data.yaml).
    Returns {id: name} or None if nothing found.
    """
    search_dirs = [labels_dir, labels_dir.parent]

    for d in search_dirs:
        classes_txt = d / "classes.txt"
        if classes_txt.exists():
            names = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
            return {i: n for i, n in enumerate(names)}

        notes_json = d / "notes.json"
        if notes_json.exists():
            try:
                data = json.loads(notes_json.read_text())
                cats = data.get("categories", [])
                # roboflow/labelImg notes.json typically: [{"id":0,"name":"board"}, ...]
                return {c["id"]: c["name"] for c in cats}
            except Exception:
                pass

        data_yaml = d / "data.yaml"
        if data_yaml.exists():
            try:
                import yaml  # optional dependency
                data = yaml.safe_load(data_yaml.read_text())
                names = data.get("names")
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                if isinstance(names, list):
                    return {i: n for i, n in enumerate(names)}
            except ImportError:
                print("  [fen] NOTE: pyyaml not installed, cannot parse data.yaml for class names "
                      "(pip install pyyaml --break-system-packages)")
            except Exception:
                pass

    return None


def resolve_excluded_class_ids(labels_dir: Path, fallback_id: int = 0) -> set[int]:
    # 1) Prefer an explicit class map found on disk next to the labels.
    class_map = load_class_names(labels_dir)
    if class_map:
        board_ids = {i for i, n in class_map.items() if n.strip().lower() == "board"}
        if board_ids:
            print(f"  [fen] Found classes.txt/notes.json/data.yaml, 'board' = id(s) "
                  f"{sorted(board_ids)} -> excluding")
            return board_ids
        print(f"  [fen] WARNING: class map found on disk but no 'board' entry in it: {class_map}")

    # 2) Fall back to the known FENiT-FEN class map (confirmed manually).
    board_ids = {i for i, n in DEFAULT_FEN_CLASS_MAP.items() if n.strip().lower() == "board"}
    if board_ids:
        print(f"  [fen] No class-map file found on disk; using built-in known FENiT-FEN class map, "
              f"'Board' = id(s) {sorted(board_ids)} -> excluding")
        return board_ids

    # 3) Last resort.
    print(f"  [fen] WARNING: could not resolve a 'board' class id from any source. "
          f"Falling back to excluding class id {fallback_id}. VERIFY this is correct.")
    return {fallback_id}


def collect_yolo_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        img_path = find_image_for_stem(images_dir, label_path.stem)
        if img_path is None:
            print(f"  [fen] WARNING: no image found for label {label_path.name}, skipping")
            continue
        pairs.append((img_path, label_path))
    return pairs


def yolo_pose_to_yolo_lines(label_path: Path, excluded_class_ids: set[int]) -> tuple[list[str], int]:
    """Returns (yolo_lines, num_boxes_dropped_as_board)."""
    lines = []
    dropped_board = 0
    for raw_line in label_path.read_text().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id in excluded_class_ids:
            dropped_board += 1
            continue
        cx, cy, w, h = (float(x) for x in parts[1:5])
        lines.append(f"{PIECE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, dropped_board


# --------------------------------------------------------------------------- #
# Output writing
# --------------------------------------------------------------------------- #

def write_split(records, out_root: Path, split: str, dry_run: bool):
    img_out_dir = out_root / "images" / split
    lbl_out_dir = out_root / "labels" / split
    if not dry_run:
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for out_stem, img_path, yolo_lines in records:
        if not dry_run:
            shutil.copy2(img_path, img_out_dir / f"{out_stem}{img_path.suffix.lower()}")
            (lbl_out_dir / f"{out_stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))


def write_data_yaml(out_root: Path, dry_run: bool):
    # Deliberately NOT writing an absolute `path:` here. Baking in a resolved
    # absolute path (e.g. C:\Users\...\organized_data\piece_detection) breaks
    # the moment this folder is copied/uploaded to another machine (Colab,
    # Drive, another PC) because Ultralytics treats that path literally.
    # With no `path:` key, Ultralytics resolves train/val relative to the
    # directory data.yaml itself lives in, which makes the whole
    # organized_data/piece_detection folder portable as-is.
    content = (
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['{PIECE_CLASS_NAME}']\n"
    )
    if not dry_run:
        (out_root / "data.yaml").write_text(content)
    else:
        print("--- data.yaml (dry-run preview) ---")
        print(content)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parent.parent  # data_organizing_scripts/.. = project root
    parser.add_argument("--project-root", type=Path, default=default_root,
                         help=f"Main project directory (default: {default_root})")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of images held out for val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    args = parser.parse_args()

    root = args.project_root
    voc_images_dir = root / "raw_data" / "Chess Piece Detection" / "images"
    voc_annots_dir = root / "raw_data" / "Chess Piece Detection" / "annotations"
    fen_images_dir = root / "raw_data" / "FENiT-FEN" / "images"
    fen_labels_dir = root / "raw_data" / "FENiT-FEN" / "labels"
    out_root = root / "organized_data" / "piece_detection"

    for d in (voc_images_dir, voc_annots_dir, fen_images_dir, fen_labels_dir):
        if not d.exists():
            print(f"ERROR: expected directory not found: {d}")
            sys.exit(1)

    all_records = []  # (out_stem, img_path, yolo_lines)
    total_pieces = 0
    total_board_dropped = 0

    # --- Source 1: VOC XML ---
    print("Processing Source 1: Chess Piece Detection (VOC XML)...")
    voc_pairs = collect_voc_pairs(voc_images_dir, voc_annots_dir)
    for img_path, xml_path in voc_pairs:
        lines, dropped = voc_xml_to_yolo_lines(xml_path, img_path)
        total_board_dropped += dropped
        if not lines:
            print(f"  [voc] WARNING: {xml_path.name} produced 0 piece boxes, skipping image")
            continue
        total_pieces += len(lines)
        all_records.append((f"voc_{img_path.stem}", img_path, lines))
    print(f"  -> {len(voc_pairs)} pairs found, {sum(1 for r in all_records if r[0].startswith('voc_'))} kept")

    # --- Source 2: YOLO-pose txt ---
    print("Processing Source 2: FENiT-FEN (YOLO-pose txt)...")
    excluded_ids = resolve_excluded_class_ids(fen_labels_dir)
    fen_pairs = collect_yolo_pairs(fen_images_dir, fen_labels_dir)
    for img_path, label_path in fen_pairs:
        lines, dropped = yolo_pose_to_yolo_lines(label_path, excluded_ids)
        total_board_dropped += dropped
        if not lines:
            print(f"  [fen] WARNING: {label_path.name} produced 0 piece boxes, skipping image")
            continue
        total_pieces += len(lines)
        all_records.append((f"fen_{img_path.stem}", img_path, lines))
    print(f"  -> {len(fen_pairs)} pairs found, {sum(1 for r in all_records if r[0].startswith('fen_'))} kept")

    if not all_records:
        print("ERROR: no usable records collected from either source, aborting.")
        sys.exit(1)

    # --- Split ---
    random.seed(args.seed)
    shuffled = all_records[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_split))
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    print("\nWriting output dataset...")
    write_split(train_records, out_root, "train", args.dry_run)
    write_split(val_records, out_root, "val", args.dry_run)
    write_data_yaml(out_root, args.dry_run)

    print("\n=== Summary ===")
    print(f"Total images kept:      {len(all_records)}  (train: {len(train_records)}, val: {len(val_records)})")
    print(f"Total piece boxes kept: {total_pieces}")
    print(f"Total board boxes dropped: {total_board_dropped}")
    print(f"Output written to: {out_root.resolve()}" + (" (dry-run, nothing actually written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
