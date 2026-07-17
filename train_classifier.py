"""
Chess OCR — Model 2 Square Classifier Training Script
=======================================================
Trains a custom lightweight CNN on the CSV splits produced by build_splits.py.

Architecture: DepthwiseSepCNN
  - 5 blocks of depthwise separable convolutions
  - Residual (skip) connections
  - Input: 64x64 grayscale (single channel)
  - Output: 13 classes (Empty + 6 white + 6 black pieces)
  - ~750K parameters
  - Exports to ONNX after training

Usage:
    python train_classifier.py
    python train_classifier.py                                      # train on synthetic (default)
    python train_classifier.py --mode physical                        # train on physical data
    python train_classifier.py --mode synthetic --epochs 40 --batch 64
    python train_classifier.py --splits ./organized_data/classification/splits/synthetic --run ./runs/synthetic
"""

import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict



import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import onnx

# ── Label maps ────────────────────────────────────────────────────────────────

LABEL_MAP_SYNTHETIC = {
    "Empty": 0,
    "wK": 1, "wQ": 2, "wR": 3, "wB": 4, "wN": 5, "wP": 6,
    "bK": 7, "bQ": 8, "bR": 9, "bB": 10, "bN": 11, "bP": 12,
}

# Physical has no Empty — 12 pieces remapped to 0-based indices
LABEL_MAP_PHYSICAL = {
    "wK": 0, "wQ": 1, "wR": 2, "wB": 3, "wN": 4, "wP": 5,
    "bK": 6, "bQ": 7, "bR": 8, "bB": 9, "bN": 10, "bP": 11,
}

# Active map and class count — set by resolve_mode() before training
LABEL_MAP   = LABEL_MAP_SYNTHETIC
NUM_CLASSES = 13


def resolve_mode(mode: str):
    """Return (label_map, idx_to_label, num_classes) for the given mode."""
    if mode == "physical":
        lmap = LABEL_MAP_PHYSICAL
    else:
        lmap = LABEL_MAP_SYNTHETIC
    return lmap, {v: k for k, v in lmap.items()}, len(lmap)


# Module-level defaults (overridden inside train() per mode)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── Architecture ──────────────────────────────────────────────────────────────

class DepthwiseSepConv(nn.Module):
    """
    Depthwise separable convolution block:
      depthwise conv (per-channel) → pointwise conv (1x1 mix) → BN → ReLU
    ~8x fewer parameters than a standard conv of the same size.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class ResidualDSBlock(nn.Module):
    """
    Two depthwise separable convs with a residual (skip) connection.
    If channel dimensions differ, a 1x1 conv projects the skip connection.
    Stabilises training and improves gradient flow through deeper layers.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSepConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSepConv(out_ch, out_ch)

        # Project skip connection if shape changes
        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.skip(x) if self.skip else x
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity, inplace=True)


class ChessPieceCNN(nn.Module):
    """
    Lightweight CNN for 64x64 grayscale chess square classification.

    Design choices (tied to project decisions):
    - Single channel input (grayscale normalised — board-theme-agnostic)
    - Depthwise separable convs → fast inference (runs 64x per board image)
    - Residual connections → stable training on small dataset
    - Aggressive Dropout(0.4) → regularises against synthetic-only overfitting
    - GlobalAvgPool → no fixed spatial assumption, handles slight crop offsets
    - ~750K parameters → small enough for CPU inference, ONNX export < 3MB

    Layer breakdown:
      Stem:    64x64x1  → 64x64x32
      Block1:  64x64x32 → 64x64x64   (residual)
      Pool:    64x64x64 → 32x32x64
      Block2:  32x32x64 → 32x32x128  (residual)
      Pool:    32x32x128 → 16x16x128
      Block3:  16x16x128 → 16x16x128 (residual)
      Block4:  16x16x128 → 16x16x256 (residual)
      Pool:    16x16x256 → 8x8x256
      Block5:  8x8x256  → 8x8x256   (residual)
      GAP:     8x8x256  → 256
      FC:      256 → 13
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()

        # Stem — standard conv to establish initial feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Progressive blocks — channels double, spatial halves via MaxPool
        self.block1 = ResidualDSBlock(32,  64)
        self.pool1  = nn.MaxPool2d(2)          # 64 → 32

        self.block2 = ResidualDSBlock(64,  128)
        self.pool2  = nn.MaxPool2d(2)          # 32 → 16

        self.block3 = ResidualDSBlock(128, 128)

        self.block4 = ResidualDSBlock(128, 256)
        self.pool3  = nn.MaxPool2d(2)          # 16 → 8

        self.block5 = ResidualDSBlock(256, 256)

        # Global average pool — collapses 8x8 → 1x1, robust to crop offsets
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.pool3(self.block4(x))
        x = self.block5(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)   # raw logits — loss fn applies softmax internally


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Dataset ───────────────────────────────────────────────────────────────────

class ChessSquareDataset(Dataset):
    """
    Reads a CSV produced by build_splits.py.
    Each row: filepath, label, label_idx, source, split

    physical samples are upweighted by physical_weight (default 3.0)
    to compensate for their smaller count vs synthetic.
    """
    def __init__(self, csv_path, transform=None, physical_weight=3.0):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

        # Per-sample weights for WeightedRandomSampler
        self.weights = self.df["source"].map(
            {"synthetic": 1.0, "physical": physical_weight}
        ).fillna(1.0).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["filepath"]).convert("L")   # grayscale
        label = int(row["label_idx"])
        if self.transform:
            img = self.transform(img)
        return img, label


def make_transforms(train=True):
    """
    Minimal transforms — heavy augmentation already done in generate_synthetic_data.py.
    Training adds small random affine for extra robustness.
    Validation/test: just resize + tensor + normalise.
    """
    norm = transforms.Normalize(mean=[0.5], std=[0.5])   # [-1, 1] range

    if train:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            norm,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            norm,
        ])


def make_loaders(splits_dir, batch_size, num_workers=2, physical_weight=3.0):
    splits = Path(splits_dir)

    train_ds = ChessSquareDataset(splits / "train.csv", make_transforms(train=True),  physical_weight)
    val_ds   = ChessSquareDataset(splits / "val.csv",   make_transforms(train=False), physical_weight)
    test_ds  = ChessSquareDataset(splits / "test.csv",  make_transforms(train=False), physical_weight)

    # WeightedRandomSampler upweights physical samples in each batch
    sampler = WeightedRandomSampler(
        weights     = train_ds.weights,
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader

# ── Class weights for loss (handles Empty imbalance) ─────────────────────────

def compute_class_weights(csv_path, device, num_classes):
    """
    Compute inverse-frequency class weights from training CSV.
    Reindexes to full class range so missing classes don't cause shape mismatches.
    """
    df = pd.read_csv(csv_path)
    counts = df["label_idx"].value_counts()
    full_counts = counts.reindex(range(num_classes), fill_value=0).values.astype(float)
    weights = 1.0 / (full_counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalise
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, epoch=None, num_epochs=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device, idx_to_label, num_classes):
    """Compute per-class accuracy on a loader — used for final test report."""
    model.eval()
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, label in zip(preds, labels):
                class_total[label.item()]   += 1
                class_correct[label.item()] += int(pred == label)

    results = {}
    for idx in range(num_classes):
        total = class_total[idx]
        acc   = class_correct[idx] / total if total > 0 else 0.0
        results[idx_to_label[idx]] = {"correct": class_correct[idx],
                                       "total": total, "acc": acc}
    return results

# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, out_path, device):
    """
    Export trained model to ONNX.
    Input: (1, 1, 64, 64) — batch=1, channels=1, 64x64
    No PyTorch needed at inference — runs via ONNX Runtime.
    """
    model.eval()
    dummy = torch.zeros(1, 1, 64, 64, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names  = ["square_crop"],
        output_names = ["logits"],
        dynamic_axes = {"square_crop": {0: "batch"}, "logits": {0: "batch"}},
        opset_version= 18,
    )
    # Validate
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ONNX model saved → {out_path}  ({size_mb:.2f} MB)")

# ── Main training function ────────────────────────────────────────────────────

def train(
    splits_dir      = "./organized_data/classification/splits/synthetic",
    run_dir         = "./runs/synthetic",
    mode            = "synthetic",
    epochs          = 40,
    batch_size      = 64,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    dropout         = 0.4,
    physical_weight = 3.0,
    num_workers     = 2,
    patience        = 8,
):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve mode-specific label map and class count ──
    label_map, idx_to_label, num_classes = resolve_mode(mode)

    print("=" * 55)
    print("Chess OCR — Model 2 Square Classifier Training")
    print("=" * 55)
    print(f"  Mode        : {mode}  ({num_classes} classes)")
    print(f"  Device      : {device}")
    print(f"  Splits      : {splits_dir}")
    print(f"  Run dir     : {run_path.resolve()}")
    print(f"  Epochs      : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Dropout     : {dropout}  |  Weight decay: {weight_decay}")
    print(f"  Phys weight : {physical_weight}x  |  Early stop patience: {patience}")

    # ── Model ──
    model = ChessPieceCNN(num_classes=num_classes, dropout=dropout).to(device)
    print(f"\n  Parameters  : {count_params(model):,}")

    # ── Data ──
    print(f"\nLoading data from {splits_dir}...")
    train_loader, val_loader, test_loader = make_loaders(
        splits_dir, batch_size, num_workers, physical_weight
    )
    print(f"  Train batches: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    # ── Loss — weighted CrossEntropy handles class imbalance ──
    class_weights = compute_class_weights(
        Path(splits_dir) / "train.csv", device, num_classes
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine annealing: lr decays smoothly to lr/10 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # ── Training loop ──
    print("\nTraining...\n")
    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}"
    print(header)
    print("-" * len(header))

    best_val_acc  = 0.0
    epochs_no_imp = 0
    history       = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            epoch=epoch, num_epochs=epochs,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
            epoch=epoch, num_epochs=epochs,
        )
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>8.2e} | {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_acc":    val_acc,
                "label_map":  label_map,
                "num_classes": num_classes,
                "mode":       mode,
            }, run_path / "best.pt")
            print(f"        ✓ New best val acc: {val_acc:.2%} — checkpoint saved")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Save training history ──
    pd.DataFrame(history).to_csv(run_path / "history.csv", index=False)

    # ── Load best and evaluate on test set ──
    print(f"\nLoading best checkpoint (val acc: {best_val_acc:.2%})...")
    ckpt = torch.load(run_path / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print(f"\nTest set results:")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  Acc  : {test_acc:.2%}")

    # Per-class accuracy report
    print(f"\nPer-class accuracy on test set:")
    per_class = per_class_accuracy(model, test_loader, device, idx_to_label, num_classes)
    for cls, res in per_class.items():
        if res["total"] == 0:
            continue   # skip classes absent from this split
        bar   = "█" * int(res["acc"] * 20)
        flag  = "  ← CHECK" if res["acc"] < 0.80 else ""
        print(f"  {cls:8s}: {res['acc']:>6.1%}  {bar}  ({res['correct']}/{res['total']}){flag}")

    # ── Export to ONNX ──
    print(f"\nExporting to ONNX...")
    export_onnx(model, run_path / "square_classifier.onnx", device)

    # ── Save label map alongside ONNX ──
    with open(run_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # ── Final summary ──
    print(f"\n{'='*55}")
    print(f"DONE")
    print(f"  Best val acc  : {best_val_acc:.2%}")
    print(f"  Test acc      : {test_acc:.2%}")
    print(f"  Checkpoint    : {run_path}/best.pt")
    print(f"  ONNX model    : {run_path}/square_classifier.onnx")
    print(f"  History CSV   : {run_path}/history.csv")
    print(f"\nNext step: python onnx_classify.py --model {run_path}/square_classifier.onnx --image <crop.png>")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the chess square classifier (Model 2)."
    )
    p.add_argument("--splits",          default=None,             help="Splits directory (overrides --mode if set)")
    p.add_argument("--run",             default=None,             help="Output directory for this run (overrides --mode if set)")
    p.add_argument("--mode",            default="synthetic",      choices=["synthetic", "physical"],
                   help="Which data source to train on — sets default --splits and --run (default: synthetic)")
    p.add_argument("--epochs",          type=int,   default=40,   help="Max training epochs (default 40)")
    p.add_argument("--batch",           type=int,   default=64,   help="Batch size (default 64)")
    p.add_argument("--lr",              type=float, default=1e-3, help="Initial learning rate (default 1e-3)")
    p.add_argument("--weight-decay",    type=float, default=1e-4, help="AdamW weight decay (default 1e-4)")
    p.add_argument("--dropout",         type=float, default=0.4,  help="Dropout rate (default 0.4)")
    p.add_argument("--physical-weight", type=float, default=3.0,  help="Upweight physical samples (default 3.0)")
    p.add_argument("--workers",         type=int,   default=2,    help="DataLoader workers (default 2)")
    p.add_argument("--patience",        type=int,   default=8,    help="Early stopping patience (default 8)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --mode sets sensible defaults; --splits/--run override if explicitly given
    SPLITS_ROOT = "./organized_data/classification/splits"
    splits_dir = args.splits if args.splits else f"{SPLITS_ROOT}/{args.mode}"
    run_dir    = args.run    if args.run    else f"./runs/{args.mode}"

    # Validate splits dir exists
    from pathlib import Path as _Path
    if not (_Path(splits_dir) / "train.csv").exists():
        print(f"\nERROR: train.csv not found in {splits_dir}")
        print(f"  Run build_splits.py first, or check --splits / --mode")
        exit(1)

    print(f"  Mode    : {args.mode}")
    print(f"  Splits  : {splits_dir}")
    print(f"  Run dir : {run_dir}")

    train(
        splits_dir      = splits_dir,
        run_dir         = run_dir,
        mode            = args.mode,
        epochs          = args.epochs,
        batch_size      = args.batch,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        dropout         = args.dropout,
        physical_weight = args.physical_weight,
        num_workers     = args.workers,
        patience        = args.patience,
    )
