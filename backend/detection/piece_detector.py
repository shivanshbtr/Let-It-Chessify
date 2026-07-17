"""
Chess OCR — Physical Piece Detector (YOLOv8 single-class)
===========================================================
Wraps the trained single-class piece detection model.
Trained with: yolov8n, imgsz=640, single_cls=True
Output: bounding boxes only — single class "piece", no type label.
Piece TYPE classification happens downstream in the CNN classifier.

Located at: models/piece_detection/train/weights/best.pt

This module only detects WHERE pieces are on the raw image.
It does NOT classify what type of piece each one is.
"""

from pathlib import Path
import numpy as np

_model_cache = {}

PIECE_DETECTION_IMGSZ = 640   # must match training imgsz


def load_piece_detector(weights_path):
    """Load (and cache) the YOLOv8 piece detection model."""
    weights_path = str(weights_path)
    if weights_path in _model_cache:
        return _model_cache[weights_path]

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Piece detection weights not found: {weights_path}\n"
            f"Expected at models/piece_detection/train/weights/best.pt"
        )

    from ultralytics import YOLO
    model = YOLO(weights_path)
    _model_cache[weights_path] = model
    print(f"[PieceDetector] Loaded: {weights_path}")
    return model


def detect_pieces(img_rgb, weights_path, conf_threshold=0.25):
    """
    Run YOLO piece detection on the raw image.

    Args:
        img_rgb:        RGB numpy array (H, W, 3) — original image
        weights_path:   path to best.pt
        conf_threshold: minimum detection confidence (default 0.25)

    Returns:
        list of dicts, one per detected piece:
        {
            "bbox":       [xmin, ymin, xmax, ymax] in pixel coords,
            "center":     (cx, cy) in pixel coords,
            "confidence": float,
        }
        Empty list if nothing detected.
    """
    model   = load_piece_detector(weights_path)
    results = model.predict(
        img_rgb,
        imgsz   = PIECE_DETECTION_IMGSZ,
        conf    = conf_threshold,
        verbose = False,
    )

    if not results or len(results) == 0:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    pieces = []
    boxes  = result.boxes.xyxy.cpu().numpy()    # (N, 4) — xmin,ymin,xmax,ymax
    confs  = result.boxes.conf.cpu().numpy()    # (N,)

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        pieces.append({
            "bbox":       [float(xmin), float(ymin), float(xmax), float(ymax)],
            "center":     (float(cx), float(cy)),
            "confidence": float(confs[i]),
        })

    return pieces
