"""
Chess OCR — Preprocessing (Two Separate, Faithful Pipelines)
================================================================
CRITICAL: synthetic and physical models were trained on DIFFERENT
preprocessing recipes. This module replicates each EXACTLY as used
in the training data generation scripts. Do not "simplify" or merge
these — that was the root cause of an earlier accuracy bug.

Synthetic recipe (from generate_synthetic_data.py):
    1. Grayscale FIRST           — crop_img.convert("L")
    2. Background normalize      — shift grayscale array towards
                                    target light/dark gray, computed
                                    from corner-patch mean
    3. CLAHE                     — applied directly on the grayscale
                                    array (clipLimit=2.0, tile=(4,4))
    4. Output                    — single-channel array, mode "L"

Physical recipe (from Chess Piece Detection_organize.py AND
FENiT-FEN_organize.py — confirmed identical in both):
    1. Crop tight bbox, NO padding, NO background normalization
       (no square context exists in a tight piece crop)
    2. CLAHE on V-channel of HSV — convert RGB->HSV, clahe.apply(V),
       convert back HSV->RGB
    3. Grayscale LAST            — convert("L") AFTER the HSV CLAHE
    4. Output                    — single-channel array, mode "L"

These are genuinely different operation orders and must stay that way.
"""

import numpy as np
from PIL import Image
import cv2

# ── Shared constants ──────────────────────────────────────────────────────────

LIGHT_SQUARE_COLOR = 200.0   # target gray value for light squares (synthetic only)
DARK_SQUARE_COLOR  = 100.0   # target gray value for dark squares  (synthetic only)
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (4, 4)


# ── SYNTHETIC preprocessing (digital pipeline) ────────────────────────────────

def preprocess_synthetic_crop(crop_img: Image.Image, file_idx: int, rank_idx: int) -> Image.Image:
    """
    Exact replica of preprocess_crop() in generate_synthetic_data.py.

    Args:
        crop_img: PIL Image, any mode, square crop from the warped board
        file_idx: 0-7 (a-h)
        rank_idx: 0-7 (rank 1-8, i.e. rank_idx=0 means rank "1")

    Returns:
        PIL Image, mode "L" (grayscale), same size as input
        (caller resizes to 64x64 separately, matching original script order)
    """
    # python-chess convention: light square when (file + rank) % 2 == 1
    is_light  = (file_idx + rank_idx) % 2 == 1
    target_bg = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR

    # Step 1 — grayscale FIRST
    gray = np.array(crop_img.convert("L"), dtype=np.float32)

    # Step 2 — background normalize using 4x4 corner patches
    corners = np.concatenate([
        gray[:4, :4].ravel(), gray[:4, -4:].ravel(),
        gray[-4:, :4].ravel(), gray[-4:, -4:].ravel(),
    ])
    shift = target_bg - corners.mean()
    gray  = np.clip(gray + shift, 0, 255).astype(np.uint8)

    # Step 3 — CLAHE directly on the grayscale array
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    gray  = clahe.apply(gray)

    return Image.fromarray(gray, mode="L")


# ── PHYSICAL preprocessing (physical pipeline) ────────────────────────────────

def preprocess_physical_crop(crop_img: Image.Image) -> Image.Image:
    """
    Exact replica of apply_clahe() in Chess_Piece_Detection_organize.py
    and FENiT-FEN_organize.py (confirmed byte-identical logic in both).

    NOTE: deliberately does NOT take file_idx/rank_idx — physical training
    data has no square-color normalization step at all (tight bbox crops
    have no square background to normalize against).

    Args:
        crop_img: PIL Image, any mode, crop from the warped board
                  (1.5x context square, per Section 21.2 of project log)

    Returns:
        PIL Image, mode "L" (grayscale), same size as input
        (caller resizes to 64x64 separately, matching original script order)
    """
    # Step 1 — ensure RGB, convert to HSV
    img_np  = np.array(crop_img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Step 2 — CLAHE on V channel only
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])

    # Step 3 — back to RGB, THEN grayscale (order matters — matches training)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_rgb).convert("L")


# ── Final resize (applied identically after either pipeline) ─────────────────

def resize_to_model_input(img: Image.Image, size: int = 64) -> Image.Image:
    """Both training scripts resize to 64x64 via LANCZOS as the final step."""
    return img.resize((size, size), Image.LANCZOS)
