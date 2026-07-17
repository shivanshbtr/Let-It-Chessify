"""
Chess OCR — Physical Board Corner Detector (YOLOv8n-pose)
=============================================================
Wraps the trained YOLOv8n-pose model (Section 21.2 / 23 of project log).
Trained on chessboard-corner-detect + FENiT-FEN datasets.
mAP50-95: 0.995 on held-out session-split validation.

kpt_shape = [4, 3]  — 4 keypoints, each (x, y, visibility)
Keypoint order (fixed at training time, see chessboard-corner-detect
_organize.py data.yaml comment):
    kpt 0 — Top-Left
    kpt 1 — Top-Right
    kpt 2 — Bottom-Left
    kpt 3 — Bottom-Right

NOTE the training data.yaml orders TL, TR, BL, BR (not TL,TR,BR,BL).
This wrapper re-orders to our pipeline's standard [TL, TR, BR, BL]
before returning, so downstream code (compute_homography, grid_from_corners)
never needs to know about the training-time keypoint order.
"""

import numpy as np
from pathlib import Path

_model_cache = {}


def load_corner_model(weights_path):
    """
    Load (and cache) the YOLOv8n-pose corner detection model.
    Lazy import of ultralytics so the rest of the backend can run
    even if ultralytics isn't installed (e.g. digital-only deployments).
    """
    weights_path = str(weights_path)
    if weights_path in _model_cache:
        return _model_cache[weights_path]

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Corner detection weights not found: {weights_path}\n"
            f"Expected at models/corner_detection/train-2/weights/best.pt "
            f"(or wherever CORNER_MODEL_PATH points)."
        )

    from ultralytics import YOLO
    model = YOLO(weights_path)
    _model_cache[weights_path] = model
    print(f"[CornerDetector] Loaded: {weights_path}")
    return model


def detect_physical_board_corners(img_np, weights_path, conf_threshold=0.5):
    """
    Run YOLOv8n-pose to find the 4 board corners on a physical board photo.

    Args:
        img_np:         RGB numpy array (H, W, 3)
        weights_path:   path to best.pt
        conf_threshold: minimum detection confidence to accept (default 0.5)

    Returns:
        corners (4x2 float32 array, ordered [TL, TR, BR, BL])
        or None if no confident detection found
        Also returns the raw detection confidence as a second value.
    """
    model   = load_corner_model(weights_path)
    results = model.predict(img_np, verbose=False, conf=conf_threshold)

    if not results or len(results) == 0:
        return None, 0.0

    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        return None, 0.0

    # Take the highest-confidence detected board (should normally be exactly one)
    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = int(result.boxes.conf.argmax())
        box_conf = float(result.boxes.conf[best_idx])
    else:
        best_idx = 0
        box_conf = float(conf_threshold)

    kpts = result.keypoints.xy[best_idx].cpu().numpy()  # shape (4, 2)

    if kpts.shape[0] < 4:
        return None, 0.0

    # Training-time order: TL, TR, BL, BR  →  reorder to TL, TR, BR, BL
    tl, tr, bl, br = kpts[0], kpts[1], kpts[2], kpts[3]
    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    return corners, box_conf
