"""
Chess OCR — Physical Data Prep Script
=======================================
Converts the Kaggle Pascal VOC XML dataset into 64x64 piece crops
for Model 2 Part 2 (physical) training data.

Input directory structure:
    raw_physical/
        images/         ← .JPG files
        annotations/    ← .xml files (same stem as image)

Output directory structure:
    data/physical/
        Empty/          ← intentionally left empty (no Empty in this dataset)
        wK/ wQ/ wR/ wB/ wN/ wP/
        bK/ bQ/ bR/ bB/ bN/ bP/

Key decisions (from project log Section 18b):
    - Tight bbox crops, NO padding — full 3D piece, no background
    - Same preprocessing as synthetic: CLAHE on V channel
    - NO square color normalization (no square background to normalize)
    - Augmentation applied: brightness, contrast, rotation, flip, noise
    - wN and bB get 3x extra augmentation (rare classes per Stanford paper)

Usage:
    python prep_physical_data.py
    python prep_physical_data.py --raw ./raw_physical --out ./data/physical
    python prep_physical_data.py --augment 6 --min-size 20
"""

import os
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ── Label mapping ─────────────────────────────────────────────────────────────
# Pascal VOC name strings → our 13-class system
# Must match LABEL_MAP in build_splits.py and train_classifier.py

VOC_TO_CLASS = {
    "white-king":   "wK",
    "white-queen":  "wQ",
    "white-rook":   "wR",
    "white-bishop": "wB",
    "white-knight": "wN",
    "white-pawn":   "wP",
    "black-king":   "bK",
    "black-queen":  "bQ",
    "black-rook":   "bR",
    "black-bishop": "bB",
    "black-knight": "bN",
    "black-pawn":   "bP",
}

CLASSES = [
    "Empty",
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]

# Classes that get extra augmentation (sparse in Kaggle — Stanford paper finding)
EXTRA_AUGMENT_CLASSES = {"wK","wQ","wR","wB","wN","bK","bB","bQ","bR","bN"}
EXTRA_AUGMENT_MULTIPLIER = 2

# ── XML parsing ───────────────────────────────────────────────────────────────

def parse_xml(xml_path):
    """
    Parse a Pascal VOC XML annotation file.
    Returns: image filename, image (w, h), list of {name, xmin, ymin, xmax, ymax}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size     = root.find("size")
    img_w    = int(size.findtext("width"))
    img_h    = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip().lower()
        if name not in VOC_TO_CLASS:
            continue  # skip unknown class names
        bbox = obj.find("bndbox")
        objects.append({
            "name":  name,
            "label": VOC_TO_CLASS[name],
            "xmin":  int(float(bbox.findtext("xmin"))),
            "ymin":  int(float(bbox.findtext("ymin"))),
            "xmax":  int(float(bbox.findtext("xmax"))),
            "ymax":  int(float(bbox.findtext("ymax"))),
        })

    return filename, (img_w, img_h), objects

# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_clahe(crop_img):
    """
    CLAHE on V channel — boosts dark piece visibility.
    Same as synthetic pipeline (Section 13 of project log).
    NOTE: No square color normalization here — physical crops have no
    square background to normalize (tight bbox crop decision, Section 18b).
    """
    img_np  = np.array(crop_img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_rgb).convert("L")  # → grayscale, matches synthetic

def crop_and_preprocess(img, xmin, ymin, xmax, ymax, min_size=20):
    """
    Crop piece bbox from image, validate size, apply CLAHE, resize to 64x64.
    Returns None if crop is too small to be useful.
    """
    img_w, img_h = img.size

    # Clamp to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    w = xmax - xmin
    h = ymax - ymin

    # Skip degenerate crops
    if w < min_size or h < min_size:
        return None

    crop = img.crop((xmin, ymin, xmax, ymax))
    crop = apply_clahe(crop)
    crop = crop.resize((64, 64), Image.LANCZOS)
    return crop

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_crop(crop_img):
    """
    Random augmentation for a single 64x64 grayscale crop.
    Same augmentation suite as synthetic pipeline for consistency.
    """
    img = crop_img.copy()

    # Brightness
    if random.random() > 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))

    # Contrast
    if random.random() > 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    # Rotation (pieces can appear at slight angles in photos)
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

# ── Save helper ───────────────────────────────────────────────────────────────

def save_crop(crop, label, out_dir, filename):
    cls_dir = out_dir / label
    cls_dir.mkdir(parents=True, exist_ok=True)
    crop.save(cls_dir / filename)

# ── Main pipeline ─────────────────────────────────────────────────────────────

def prep_physical_data(
    raw_dir   = "./../raw_data/Chess Piece Detection",
    out_dir   = "./../organized_data/classification/physical",
    augment_n = 6,
    min_size  = 20,
    seed      = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    images_dir      = raw_path / "images"
    annotations_dir = raw_path / "annotations"

    # Validate input structure
    if not images_dir.exists():
        print(f"ERROR: images/ folder not found at {images_dir}")
        return
    if not annotations_dir.exists():
        print(f"ERROR: annotations/ folder not found at {annotations_dir}")
        return

    # Create output class dirs
    for cls in CLASSES:
        (out_path / cls).mkdir(parents=True, exist_ok=True)

    # Find all XML files
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        print(f"ERROR: No XML files found in {annotations_dir}")
        return

    print("=" * 55)
    print("Chess OCR — Physical Data Prep")
    print("=" * 55)
    print(f"Input : {raw_path.resolve()}")
    print(f"Output: {out_path.resolve()}")
    print(f"XMLs  : {len(xml_files)} annotation files found")
    print(f"Aug   : {augment_n} per crop (x{EXTRA_AUGMENT_MULTIPLIER} for wN, bB)")
    print()

    class_counts  = defaultdict(int)
    skipped_imgs  = 0
    skipped_crops = 0
    total_saved   = 0
    crop_idx      = 0

    for xml_path in xml_files:
        # Find matching image
        filename, (img_w, img_h), objects = parse_xml(xml_path)

        # Try multiple extensions
        img_path = None
        for ext in [".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"]:
            candidate = images_dir / (xml_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        # Also try the filename from XML
        if img_path is None:
            candidate = images_dir / filename
            if candidate.exists():
                img_path = candidate

        if img_path is None:
            print(f"  SKIP: no image found for {xml_path.name}")
            skipped_imgs += 1
            continue

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  SKIP: failed to load {img_path.name} — {e}")
            skipped_imgs += 1
            continue

        # Verify size matches XML
        actual_w, actual_h = img.size
        if actual_w != img_w or actual_h != img_h:
            # Rescale bbox coords proportionally
            scale_x = actual_w / img_w
            scale_y = actual_h / img_h
            for obj in objects:
                obj["xmin"] = int(obj["xmin"] * scale_x)
                obj["xmax"] = int(obj["xmax"] * scale_x)
                obj["ymin"] = int(obj["ymin"] * scale_y)
                obj["ymax"] = int(obj["ymax"] * scale_y)

        # Process each annotated piece
        for obj in objects:
            label = obj["label"]

            crop = crop_and_preprocess(
                img,
                obj["xmin"], obj["ymin"],
                obj["xmax"], obj["ymax"],
                min_size=min_size,
            )

            if crop is None:
                skipped_crops += 1
                continue

            # Save base crop
            fname = f"phy_{crop_idx:06d}_base.png"
            save_crop(crop, label, out_path, fname)
            class_counts[label] += 1
            total_saved += 1

            # Augmented copies
            n_aug = augment_n
            if label in EXTRA_AUGMENT_CLASSES:
                n_aug = augment_n * EXTRA_AUGMENT_MULTIPLIER

            for aug_i in range(n_aug):
                aug = augment_crop(crop)
                fname = f"phy_{crop_idx:06d}_aug{aug_i}.png"
                save_crop(aug, label, out_path, fname)
                class_counts[label] += 1
                total_saved += 1

            crop_idx += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Images processed : {len(xml_files) - skipped_imgs}")
    print(f"Images skipped   : {skipped_imgs}")
    print(f"Crops skipped    : {skipped_crops} (too small < {min_size}px)")
    print(f"Total saved      : {total_saved}")
    print()

    print("Per-class counts:")
    max_count = max(class_counts.values()) if class_counts else 1
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        bar   = "█" * int(40 * count / max(max_count, 1))
        flag  = "  ← RARE" if 0 < count < 50 else ("  ← EMPTY (no physical labels)" if count == 0 and cls != "Empty" else "")
        print(f"  {cls:8s}: {count:5d}  {bar}{flag}")

    print()
    piece_counts = [class_counts.get(c, 0) for c in CLASSES if c != "Empty"]
    if piece_counts:
        min_c = min(piece_counts)
        max_c = max(piece_counts)
        print(f"Piece class ratio: {max_c/max(min_c,1):.1f}x  (min:{min_c} max:{max_c})")
        if max_c / max(min_c, 1) > 8:
            print("WARNING: High imbalance — consider extra augmentation on rare classes")
            print("         Training script handles this via class_weight in CrossEntropyLoss")

    print()
    print(f"Physical data ready at: {out_path.resolve()}")
    print(f"Next step: python build_splits.py  (merges synthetic + physical)")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Prep Kaggle Pascal VOC physical chess dataset into 64x64 piece crops."
    )
    p.add_argument("--raw",      default="./../raw_data/Chess Piece Detection",  help="Raw dataset root (contains images/ and annotations/)")
    p.add_argument("--out",      default="./../organized_data/classification/physical", help="Output directory")
    p.add_argument("--augment",  type=int,   default=6,     help="Augmented copies per crop (default 6)")
    p.add_argument("--min-size", type=int,   default=20,    help="Min bbox px to include (default 20)")
    p.add_argument("--seed",     type=int,   default=42,    help="Random seed (default 42)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prep_physical_data(
        raw_dir   = args.raw,
        out_dir   = args.out,
        augment_n = args.augment,
        min_size  = args.min_size,
        seed      = args.seed,
    )
