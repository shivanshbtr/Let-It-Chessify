#!/usr/bin/env python3
"""
train_piece_detection.py

Trains the single-class ("piece") chess piece detector on the dataset
produced by data_organizing_scripts/organize_piece_detection_data.py.

Lives at: main_project_dir/train_piece_detection.py
Expects:  main_project_dir/organized_data/piece_detection/data.yaml

This is a *detector only* (bounding boxes, single class "piece"). Piece
type classification happens downstream in your separate classifier model,
so this script deliberately refuses to train if data.yaml declares more
than 1 class -- that would mean the organizing step wasn't run/updated
correctly.

Usage:
    python train_piece_detection.py
    python train_piece_detection.py --model-size s --epochs 150 --imgsz 800
    python train_piece_detection.py --resume
    python train_piece_detection.py --export-onnx
    python train_piece_detection.py --device 0 --batch 16 --workers 8

Requires: pip install ultralytics --break-system-packages
"""

import argparse
import os
import sys
from pathlib import Path


def check_ultralytics():
    try:
        import ultralytics  # noqa: F401
        return ultralytics
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("Install it with: pip install ultralytics --break-system-packages")
        sys.exit(1)


def check_data_yaml(data_yaml: Path):
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run data_organizing_scripts/organize_piece_detection_data.py first.")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml is not installed. Install with: pip install pyyaml --break-system-packages")
        sys.exit(1)

    cfg = yaml.safe_load(data_yaml.read_text())
    nc = cfg.get("nc")
    names = cfg.get("names")
    if nc != 1:
        print(f"ERROR: {data_yaml} declares nc={nc}, expected nc=1 for a single-class piece detector.")
        print(f"       names={names}")
        print("This script only trains a generic piece detector -- piece-type classification is a "
              "separate downstream model. Re-run the data organizing script if this looks wrong.")
        sys.exit(1)

    print(f"Verified data.yaml: nc=1, names={names}")
    return cfg


def report_environment():
    import torch
    print("--- Environment ---")
    print(f"PyTorch:        {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU:            {props.name} ({total_gb:.1f} GB)")
    else:
        print("GPU:            none detected -- training will run on CPU and will be slow. "
              "If you're on Colab, make sure Runtime > Change runtime type > GPU is selected.")
    print(f"CPU cores:      {os.cpu_count()}")
    print("-------------------")
    return cuda_available


def resolve_resume_checkpoint(project: Path, name: str) -> str | None:
    last_ckpt = project / name / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        return str(last_ckpt)
    print(f"WARNING: --resume passed but no checkpoint found at {last_ckpt}. Starting fresh instead.")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project_root = Path(__file__).resolve().parent

    parser.add_argument("--data", type=Path,
                         default=project_root / "organized_data" / "piece_detection" / "data.yaml",
                         help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=None,
                         help="Explicit model checkpoint/config to start from "
                              "(overrides --model-size, e.g. a .pt to fine-tune further)")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                         help="YOLOv8 size if --model is not given (yolov8<size>.pt). Default: n (nano, fast)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                         help="Batch size. -1 = Ultralytics auto-batch (fits to available GPU memory)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8),
                         help="Dataloader workers")
    parser.add_argument("--device", type=str, default="",
                         help="'' = auto, '0' = first GPU, 'cpu' = force CPU, '0,1' = multi-GPU")
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "false"],
                         help="Cache images for faster epochs. Use 'disk' if dataset doesn't fit in RAM.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    parser.add_argument("--project", type=Path, default=project_root / "models" / "piece_detection",
                         help="Where to save run outputs")
    parser.add_argument("--name", type=str, default="train", help="Run name (subfolder under --project)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume the last run under --project/--name")
    parser.add_argument("--export-onnx", action="store_true",
                         help="Export best.pt to ONNX after training completes")
    args = parser.parse_args()

    ultralytics = check_ultralytics()
    from ultralytics import YOLO

    check_data_yaml(args.data)
    report_environment()

    cache = False if args.cache == "false" else args.cache

    if args.resume:
        ckpt = resolve_resume_checkpoint(args.project, args.name)
        model = YOLO(ckpt) if ckpt else YOLO(args.model or f"yolov8{args.model_size}.pt")
        resume_flag = bool(ckpt)
    else:
        model_source = args.model or f"yolov8{args.model_size}.pt"
        print(f"Loading model: {model_source}")
        model = YOLO(model_source)
        resume_flag = False

    print(f"\nStarting training -> {args.project / args.name}")
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=cache,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        resume=resume_flag,
        exist_ok=True,
        single_cls=True,  # belt-and-suspenders: force single-class training regardless of label content
    )

    run_dir = Path(model.trainer.save_dir)
    best_ckpt = run_dir / "weights" / "best.pt"
    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")

    print("\nRunning validation on best checkpoint...")
    val_model = YOLO(str(best_ckpt))
    metrics = val_model.val(data=str(args.data))
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    if args.export_onnx:
        print("\nExporting to ONNX...")
        onnx_path = val_model.export(format="onnx")
        print(f"ONNX model: {onnx_path}")

    print(f"\nDone. Best weights: {best_ckpt}")


if __name__ == "__main__":
    main()
