"""
train_corner_detection.py
==========================
Trains a YOLOv8s-pose model to detect the 4 corners
of a chessboard from a photo.

Run this on Google Colab after:
  1. Mounting Google Drive
  2. Unzipping Chess-OCR/ project folder
  3. Running corner_detection_data_organize.py (if not already done)

Usage (in Colab cell):
    !python train_corner_detection.py

Or with overrides:
    !python train_corner_detection.py --epochs 200 --batch 8
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Project root = wherever this script lives (likely a Google Drive mount in Colab)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Local (fast, non-Drive) working directory used for training when --local-copy is set.
# Reading images off a Drive FUSE mount every epoch is the #1 cause of GPU starvation
# on Colab — copying once to local disk avoids re-paying that cost every epoch.
LOCAL_ROOT = Path("/content/corner_detection_local")

# Dataset yaml (created by corner_detection_data_organize.py)
# path inside yaml is relative to PROJECT_ROOT
DATA_YAML = PROJECT_ROOT / "organized_data" / "corner_detection" / "data.yaml"

# Where to save trained weights
SAVE_DIR = PROJECT_ROOT / "models" / "corner_detection"

# ── Training hyperparameters ──────────────────
DEFAULT_CONFIG = {
    "model"    : "yolov8n-pose.pt",  # nano pose model, pretrained on COCO
    "epochs"   : 150,
    "imgsz"    : 640,
    "batch"    : 64,                 # nano-pose @640 leaves tons of headroom on a T4 (15GB)
    "lr0"      : 0.01,               # initial learning rate
    "lrf"      : 0.01,               # final lr = lr0 * lrf
    "momentum" : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience" : 30,                 # early stopping — stops if no improvement for 30 epochs
    "workers"  : 8,                  # bumped for cached data; lower if Colab gives you fewer cores
    "device"   : 0,                  # GPU 0 (Colab T4); set "cpu" if no GPU
    "seed"     : 42,
    "verbose"  : True,
    "cache"    : "ram",              # cache decoded images in RAM after first epoch — avoids re-reading Drive every epoch
}

# ── Augmentation (light — data already diverse) ──
AUGMENT_CONFIG = {
    "hsv_h"    : 0.015,   # hue shift
    "hsv_s"    : 0.4,     # saturation
    "hsv_v"    : 0.3,     # brightness
    "degrees"  : 5.0,     # rotation ±5°
    "translate": 0.1,     # translation
    "scale"    : 0.2,     # zoom
    "shear"    : 2.0,     # shear
    "flipud"   : 0.0,     # no vertical flip (board orientation matters)
    "fliplr"   : 0.5,     # horizontal flip OK
    "mosaic"   : 0.5,     # mosaic augmentation
    "mixup"    : 0.0,     # no mixup for keypoints
}


# ─────────────────────────────────────────────
# STEP 1 — Parse arguments
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8s-pose for chessboard corner detection")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch",   type=int,   default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--imgsz",   type=int,   default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--device",              default=None,
                         help="cuda device, e.g. 0, or 'cpu'. If omitted, auto-detected.")
    parser.add_argument("--resume",  action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--yes", "-y", action="store_true",
                         help="Skip the 'continue on CPU?' confirmation prompt")
    parser.add_argument("--local-copy", action="store_true",
                         help="Copy organized_data/ from Drive to local Colab disk "
                              f"({LOCAL_ROOT}) before training. Strongly recommended on "
                              "Colab — Drive's FUSE mount is slow and starves the GPU.")
    parser.add_argument("--cache", default=DEFAULT_CONFIG["cache"], choices=["ram", "disk", "False"],
                         help="Image caching mode passed to Ultralytics (default: ram). "
                              "Use 'disk' if dataset doesn't fit in RAM, 'False' to disable.")
    return parser.parse_args()


def resolve_device(args):
    """Pick a device: respect --device if given, otherwise auto-detect GPU vs CPU.
    If falling back to CPU, warn the user and ask for confirmation (unless --yes)."""
    if args.device is not None:
        return args.device

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if has_gpu:
        return 0

    print("\n" + "!" * 55)
    print("  [WARNING] No GPU detected — falling back to CPU.")
    print("  Training a YOLOv8-pose model on CPU is SLOW.")
    print("  Expect this to take much longer than on a GPU")
    print("  (potentially hours per epoch depending on dataset size).")
    print("  Consider reducing --epochs, --imgsz, and --batch,")
    print("  or running this on Google Colab with a GPU instead.")
    print("!" * 55)

    if not args.yes:
        resp = input("\n  Continue training on CPU? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("\n  Aborted by user.")
            sys.exit(0)

    return "cpu"


def maybe_copy_to_local(args):
    """If --local-copy is set, mirror organized_data/ from (likely Drive-mounted)
    PROJECT_ROOT to local Colab disk, and repoint DATA_YAML at the local copy.

    This is the single biggest lever for GPU utilization on Colab: reading
    thousands of images per epoch over the Drive FUSE mount is slow enough
    that the GPU sits idle waiting on data, regardless of batch size or
    caching settings. A one-time local copy + RAM caching removes that
    bottleneck almost entirely.
    """
    global DATA_YAML

    if not args.local_copy:
        return

    src_data_dir = PROJECT_ROOT / "organized_data"
    dst_data_dir = LOCAL_ROOT / "organized_data"

    print("\n" + "=" * 55)
    print("  Copying dataset to local disk for fast I/O...")
    print(f"  Source : {src_data_dir}")
    print(f"  Dest   : {dst_data_dir}")
    print("=" * 55)

    if dst_data_dir.exists():
        print("[OK] Local copy already exists — skipping copy "
              "(delete it manually if the source dataset changed).")
    else:
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_data_dir, dst_data_dir)
        print("[OK] Copy complete.")

    new_yaml = dst_data_dir / "corner_detection" / "data.yaml"
    assert new_yaml.exists(), f"[ERROR] Expected copied data.yaml not found at: {new_yaml}"

    # Repoint the yaml's "path:" field (if present) at the local copy, since it's
    # commonly written relative to PROJECT_ROOT and would otherwise still resolve
    # back to the slow Drive location.
    try:
        import yaml
        with open(new_yaml) as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and "path" in cfg:
            cfg["path"] = str(dst_data_dir / "corner_detection")
            with open(new_yaml, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[OK] data.yaml 'path' repointed to local copy.")
    except ImportError:
        print("[WARN] pyyaml not available — skipped rewriting data.yaml 'path' field. "
              "If training still reads from Drive, edit data.yaml manually.")

    DATA_YAML = new_yaml
    print(f"[OK] Using local data.yaml: {DATA_YAML}")


# ─────────────────────────────────────────────
# STEP 2 — Pre-flight checks
# ─────────────────────────────────────────────

def preflight(args):
    print("\n" + "=" * 55)
    print("  Corner Detection — Pre-flight Checks")
    print("=" * 55)

    # Check data.yaml exists
    assert DATA_YAML.exists(), (
        f"\n[ERROR] data.yaml not found at: {DATA_YAML}\n"
        f"  → Run corner_detection_data_organize.py first!"
    )
    print(f"[OK] data.yaml found")

    # Check train/val folders have images (relative to wherever data.yaml actually is —
    # this may be the local copy if --local-copy was used)
    data_dir = DATA_YAML.parent
    train_imgs = list((data_dir / "images" / "train").glob("*.jpg"))
    val_imgs   = list((data_dir / "images" / "val").glob("*.jpg"))

    assert len(train_imgs) > 0, "[ERROR] No training images found!"
    assert len(val_imgs)   > 0, "[ERROR] No validation images found!"

    print(f"[OK] Train images : {len(train_imgs)}")
    print(f"[OK] Val   images : {len(val_imgs)}")

    # Check ultralytics installed
    try:
        import ultralytics
        print(f"[OK] Ultralytics  : v{ultralytics.__version__}")
    except ImportError:
        print("[ERROR] ultralytics not installed!")
        print("  → Run: pip install ultralytics")
        sys.exit(1)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU          : {gpu} ({mem:.1f} GB)")
        else:
            print("[WARN] No GPU found — training on CPU will be very slow!")
            print("       In Colab: Runtime → Change runtime type → T4 GPU")
    except ImportError:
        print("[WARN] torch not found — GPU check skipped")

    # Create save directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Save dir     : {SAVE_DIR}")

    print("=" * 55)


# ─────────────────────────────────────────────
# STEP 3 — Train
# ─────────────────────────────────────────────

def train(args):
    from ultralytics import YOLO

    print("\n  Loading model...")

    if args.resume:
        # Resume from last checkpoint
        last_ckpt = SAVE_DIR / "train" / "weights" / "last.pt"
        assert last_ckpt.exists(), f"[ERROR] No checkpoint to resume from: {last_ckpt}"
        model = YOLO(str(last_ckpt))
        print(f"[OK] Resuming from: {last_ckpt}")
    else:
        # Start fresh from pretrained COCO pose weights
        model = YOLO(DEFAULT_CONFIG["model"])
        print(f"[OK] Starting from pretrained: {DEFAULT_CONFIG['model']}")

    print("\n  Starting training...\n")

    results = model.train(
        # ── Data ──────────────────────────────
        data       = str(DATA_YAML),
        project    = str(SAVE_DIR),
        name       = "train",
        exist_ok   = args.resume,       # don't error if folder exists on resume

        # ── Core ──────────────────────────────
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = args.device,
        workers    = DEFAULT_CONFIG["workers"],
        seed       = DEFAULT_CONFIG["seed"],
        verbose    = DEFAULT_CONFIG["verbose"],
        cache      = (False if args.cache == "False" else args.cache),

        # ── Learning rate ──────────────────────
        lr0        = DEFAULT_CONFIG["lr0"],
        lrf        = DEFAULT_CONFIG["lrf"],
        momentum   = DEFAULT_CONFIG["momentum"],
        weight_decay = DEFAULT_CONFIG["weight_decay"],
        warmup_epochs = DEFAULT_CONFIG["warmup_epochs"],

        # ── Early stopping ─────────────────────
        patience   = DEFAULT_CONFIG["patience"],

        # ── Augmentation ──────────────────────
        hsv_h      = AUGMENT_CONFIG["hsv_h"],
        hsv_s      = AUGMENT_CONFIG["hsv_s"],
        hsv_v      = AUGMENT_CONFIG["hsv_v"],
        degrees    = AUGMENT_CONFIG["degrees"],
        translate  = AUGMENT_CONFIG["translate"],
        scale      = AUGMENT_CONFIG["scale"],
        shear      = AUGMENT_CONFIG["shear"],
        flipud     = AUGMENT_CONFIG["flipud"],
        fliplr     = AUGMENT_CONFIG["fliplr"],
        mosaic     = AUGMENT_CONFIG["mosaic"],
        mixup      = AUGMENT_CONFIG["mixup"],

        # ── Saving ────────────────────────────
        save       = True,
        save_period= 10,               # save checkpoint every 10 epochs
    )

    return results


# ─────────────────────────────────────────────
# STEP 4 — Post-training: copy best weights
# ─────────────────────────────────────────────

def save_best_weights():
    best_src = SAVE_DIR / "train" / "weights" / "best.pt"
    best_dst = SAVE_DIR / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n[OK] Best weights copied to: {best_dst}")
    else:
        print(f"\n[WARN] best.pt not found at expected location: {best_src}")


# ─────────────────────────────────────────────
# STEP 5 — Quick validation report
# ─────────────────────────────────────────────

def validate(args):
    from ultralytics import YOLO

    best_pt = SAVE_DIR / "best.pt"
    if not best_pt.exists():
        print("[WARN] Skipping validation — best.pt not found")
        return

    print("\n" + "=" * 55)
    print("  Running validation on best.pt...")
    print("=" * 55)

    model   = YOLO(str(best_pt))
    metrics = model.val(
        data   = str(DATA_YAML),
        imgsz  = args.imgsz,
        device = args.device,
    )

    print("\n  ── Validation Metrics ──────────────────")
    print(f"  Box mAP50       : {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Keypoint mAP50  : {metrics.pose.map50:.4f}")
    print(f"  Keypoint mAP50-95: {metrics.pose.map:.4f}")
    print("  ────────────────────────────────────────")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    args.device = resolve_device(args)

    maybe_copy_to_local(args)
    preflight(args)
    train(args)
    save_best_weights()
    validate(args)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Best weights : {SAVE_DIR / 'best.pt'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
