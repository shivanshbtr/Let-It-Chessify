"""
Chess OCR — Dataset Split Builder
===================================
Scans data/classification/synthetic/ and data/classification/physical/ (if present),
and produces SEPARATE stratified train/val/test CSVs for each source.

Output structure:
    data/splits/
        synthetic/
            train.csv
            val.csv
            test.csv
        physical/               ← only created if physical data exists
            train.csv
            val.csv
            test.csv
        label_map.json          ← shared, same for both

Why separate splits?
  - Synthetic and physical have very different distributions and quantities
  - Keeps evaluation clean — you can measure model performance on each independently
  - When physical data is added later, just re-run this script — synthetic splits untouched
  - train_classifier.py can be pointed at either split dir, or both can be combined manually

Usage:
    python build_splits.py
    python build_splits.py --data ./data/classification --out ./data/splits
    python build_splits.py --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

import csv
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

# ── Label map (fixed order — never change this) ───────────────────────────────

LABEL_MAP_SYNTHETIC = {
    "Empty": 0,
    "wK": 1, "wQ": 2, "wR": 3, "wB": 4, "wN": 5, "wP": 6,
    "bK": 7, "bQ": 8, "bR": 9, "bB": 10, "bN": 11, "bP": 12,
}

# Physical data has no Empty squares — 12 classes remapped to 0-based indices
LABEL_MAP_PHYSICAL = {
    "wK": 0, "wQ": 1, "wR": 2, "wB": 3, "wN": 4, "wP": 5,
    "bK": 6, "bQ": 7, "bR": 8, "bB": 9, "bN": 10, "bP": 11,
}

# Keep LABEL_MAP as an alias for synthetic (backwards compatibility)
LABEL_MAP = LABEL_MAP_SYNTHETIC

CLASSES_SYNTHETIC = list(LABEL_MAP_SYNTHETIC.keys())
CLASSES_PHYSICAL  = list(LABEL_MAP_PHYSICAL.keys())
CLASSES = CLASSES_SYNTHETIC   # default

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_base_key(filename: str) -> str:
    """
    Strip _base / _aug<N> suffix to get the group key.
    All augmentations of the same base image land in the same split.

    Examples:
        alpha_wp_light_base.png   →  "alpha_wp_light"
        alpha_wp_light_aug03.png  →  "alpha_wp_light"
        physical_img023.png       →  "physical_img023"
    """
    stem = Path(filename).stem
    if stem.endswith("_base"):
        return stem[:-5]
    for i in range(100):
        suffix = f"_aug{i:02d}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
        # also handle _aug0 … _aug9 without zero-padding
        suffix = f"_aug{i}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


# Common alternate folder names people use for each class.
# Add to these lists if your data uses a different naming convention.
CLASS_ALIASES = {
    "Empty": ["empty", "blank", "empty_square", "empty-square", "none", "background", "bg"],
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_class_dir(source_dir: Path, cls: str) -> Path | None:
    """
    Find the folder for a class, tolerating case differences and common
    naming variants (e.g. 'empty' / 'EMPTY' / 'empty_square' / 'blank').
    """
    exact = source_dir / cls
    if exact.exists():
        return exact

    candidates = {_normalize(cls)} | {_normalize(a) for a in CLASS_ALIASES.get(cls, [])}

    for child in source_dir.iterdir():
        if child.is_dir() and _normalize(child.name) in candidates:
            return child
    return None


def scan_source(source_dir: Path, source_name: str, label_map: dict, classes: list) -> list[dict]:
    """
    Scan a source directory (synthetic/ or physical/).
    Returns list of records: {filepath, label, label_idx, source, base_key}

    label_map and classes are passed in so physical uses its own 12-class map.
    """
    if not source_dir.exists():
        return []

    records = []
    for cls in classes:
        cls_dir = _resolve_class_dir(source_dir, cls)
        if cls_dir is None:
            print(f"    !! WARNING [{source_name}]: no folder found for class '{cls}' "
                  f"under {source_dir.resolve()} — this class will have 0 images.")
            continue

        img_files = []
        for pattern in IMAGE_EXTS:
            img_files.extend(cls_dir.glob(pattern))
        img_files = sorted(set(img_files))

        if not img_files:
            print(f"    !! WARNING [{source_name}]: folder '{cls_dir.name}' exists but "
                  f"contains no matching images (checked {IMAGE_EXTS}).")

        for img_file in img_files:
            records.append({
                "filepath":  str(img_file),
                "label":     cls,
                "label_idx": label_map[cls],
                "source":    source_name,
                "base_key":  f"{cls}__{get_base_key(img_file.name)}",
            })
    return records


def stratified_split(
    records:     list[dict],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> tuple[list, list, list]:
    """
    Stratified split by class.
    Splitting is done on BASE GROUPS so augmentations never leak across splits.
    """
    rng = random.Random(seed)

    # Build base groups
    groups: dict[str, list] = defaultdict(list)
    for rec in records:
        groups[rec["base_key"]].append(rec)

    # Group by label for stratification
    label_groups: dict[str, list] = defaultdict(list)
    for base_key, group_recs in groups.items():
        label = group_recs[0]["label"]
        label_groups[label].append(group_recs)

    train_recs, val_recs, test_recs = [], [], []

    for label, class_groups in label_groups.items():
        rng.shuffle(class_groups)
        n       = len(class_groups)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))

        for g in class_groups[:n_train]:
            train_recs.extend(g)
        for g in class_groups[n_train : n_train + n_val]:
            val_recs.extend(g)
        for g in class_groups[n_train + n_val :]:
            test_recs.extend(g)

    return train_recs, val_recs, test_recs


def write_csv(records: list[dict], path: Path, split_name: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "label_idx", "source", "split"]
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "filepath":  rec["filepath"],
                "label":     rec["label"],
                "label_idx": rec["label_idx"],
                "source":    rec["source"],
                "split":     split_name,
            })
    return len(records)


def print_split_stats(split_name: str, records: list[dict], classes: list):
    counts = defaultdict(int)
    for rec in records:
        counts[rec["label"]] += 1
    total = len(records)
    print(f"\n    {split_name} ({total:,} images):")
    for cls in classes:
        n   = counts[cls]
        bar = "█" * min(n // 10, 30)
        print(f"      {cls:8s}: {n:5d}  {bar}")


def process_source(
    source_dir:  Path,
    source_name: str,
    out_dir:     Path,
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> int:
    """
    Scan, split, and write CSVs for one source (synthetic or physical).
    Returns total image count, or 0 if source doesn't exist.
    Physical uses LABEL_MAP_PHYSICAL (12 classes, 0-based, no Empty).
    Synthetic uses LABEL_MAP_SYNTHETIC (13 classes, Empty=0).
    """
    if source_name == "physical":
        label_map = LABEL_MAP_PHYSICAL
        classes   = CLASSES_PHYSICAL
    else:
        label_map = LABEL_MAP_SYNTHETIC
        classes   = CLASSES_SYNTHETIC

    records = scan_source(source_dir, source_name, label_map, classes)

    if not records:
        print(f"\n  [{source_name}] — not found or empty, skipping")
        return 0

    print(f"\n  [{source_name}]  {len(records):,} images found at {source_dir.resolve()}")

    train_recs, val_recs, test_recs = stratified_split(
        records, train_ratio, val_ratio, seed
    )

    split_out = out_dir / source_name
    n_train = write_csv(train_recs, split_out / "train.csv", "train")
    n_val   = write_csv(val_recs,   split_out / "val.csv",   "val")
    n_test  = write_csv(test_recs,  split_out / "test.csv",  "test")

    print(f"    train : {n_train:,}  →  {split_out/'train.csv'}")
    print(f"    val   : {n_val:,}  →  {split_out/'val.csv'}")
    print(f"    test  : {n_test:,}  →  {split_out/'test.csv'}")

    print_split_stats("train", train_recs, classes)
    print_split_stats("val",   val_recs,   classes)
    print_split_stats("test",  test_recs,  classes)

    return len(records)

# ── Main ──────────────────────────────────────────────────────────────────────

def build_splits(
    data_dir:    str   = "./organized_data/classification",
    out_dir:     str   = "./organized_data/classification/splits",
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    test_ratio:  float = 0.1,
    seed:        int   = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    data_path = Path(data_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("Chess OCR — Dataset Split Builder")
    print("=" * 56)
    print(f"\n  Data root : {data_path.resolve()}")
    print(f"  Out dir   : {out_path.resolve()}")
    print(f"  Ratios    : train={train_ratio:.0%}  val={val_ratio:.0%}  test={test_ratio:.0%}")
    print(f"  Seed      : {seed}")

    # ── Process each source independently ────────────────────────────────────
    syn_total = process_source(
        source_dir  = data_path / "synthetic",
        source_name = "synthetic",
        out_dir     = out_path,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        seed        = seed,
    )

    phy_total = process_source(
        source_dir  = data_path / "physical",
        source_name = "physical",
        out_dir     = out_path,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        seed        = seed,
    )

    if syn_total == 0 and phy_total == 0:
        print("\nERROR: No images found in either source. Check your data directory.")
        return

    # ── Shared label maps ─────────────────────────────────────────────────────
    label_map_path = out_path / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(LABEL_MAP_SYNTHETIC, f, indent=2)

    label_map_physical_path = out_path / "label_map_physical.json"
    with open(label_map_physical_path, "w") as f:
        json.dump(LABEL_MAP_PHYSICAL, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("DONE")
    print(f"{'='*56}")
    if syn_total:
        print(f"  organized_data/classification/splits/synthetic/train.csv")
        print(f"  organized_data/classification/splits/synthetic/val.csv")
        print(f"  organized_data/classification/splits/synthetic/test.csv")
    if phy_total:
        print(f"  organized_data/classification/splits/physical/train.csv")
        print(f"  organized_data/classification/splits/physical/val.csv")
        print(f"  organized_data/classification/splits/physical/test.csv")
    print(f"  organized_data/classification/splits/label_map.json")
    print(f"\nNext steps:")
    if syn_total:
        print(f"  python train_classifier.py --splits {out_path}/synthetic")
    if phy_total:
        print(f"  python train_classifier.py --splits {out_path}/physical")
    print(f"{'='*56}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build separate train/val/test splits for synthetic and physical data."
    )
    p.add_argument("--data",  default="./organized_data/classification",
                   help="Root containing synthetic/ and physical/ (default: ./organized_data/classification)")
    p.add_argument("--out",   default="./organized_data/classification/splits",
                   help="Output directory for split CSVs (default: ./organized_data/classification/splits)")
    p.add_argument("--train", type=float, default=0.8,  help="Train ratio (default 0.8)")
    p.add_argument("--val",   type=float, default=0.1,  help="Val ratio (default 0.1)")
    p.add_argument("--test",  type=float, default=0.1,  help="Test ratio (default 0.1)")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed (default 42)")
    args = p.parse_args()

    build_splits(
        data_dir    = args.data,
        out_dir     = args.out,
        train_ratio = args.train,
        val_ratio   = args.val,
        test_ratio  = args.test,
        seed        = args.seed,
    )
