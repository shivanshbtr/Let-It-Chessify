"""
Chess OCR — Board Mapper (Homography + Square Assignment)
===========================================================
Given 4 board corners (from YOLO corner detector), computes the
homography H that maps raw image coordinates → normalized 8x8 board space.

Then for each detected piece center (cx, cy) in raw image coords:
  1. Apply H → (bx, by) in 0–8 range (Option B — direct 8x8 mapping)
  2. col = int(floor(bx)), row = int(floor(by))  → one of 64 squares
  3. Resolve conflicts (multiple pieces → same square) by keeping
     the highest detector confidence (Section Q5 decision)

Board coordinate convention (matches python-chess):
  - Row 0 = rank 8 (top of board as seen from white's side)
  - Col 0 = file a (left side)
  - (row, col) → chess.square(col, 7 - row)

Orientation note:
  The homography is computed assuming the 4 corners are ordered
  [TL, TR, BR, BL] as returned by corner_detector.py. "TL" means
  top-left in the IMAGE, which may or may not be a8 depending on
  board orientation. Orientation correction (flip/rotate) is handled
  by the frontend "Flip board" button — not auto-detected here.
"""

import cv2
import numpy as np
import chess


def compute_homography(corners_np):
    """
    Compute homography H: raw image coords → normalized 8x8 board space.

    corners_np: (4, 2) float32 array ordered [TL, TR, BR, BL]
                as returned by detect_physical_board_corners()

    Returns: H (3x3 homography matrix, float64)

    Option B chosen (Section Q2): map directly to 8x8 space so that
    floor(transformed_x) and floor(transformed_y) give file and rank
    directly — no separate division step, no rounding ambiguity.
    """
    # Destination corners in 8x8 board space
    dst = np.array([
        [0.0, 0.0],   # TL → (file=0, rank=8 top) → board (0,0)
        [8.0, 0.0],   # TR → (file=8, rank=8 top) → board (8,0)
        [8.0, 8.0],   # BR → (file=8, rank=0 bot) → board (8,8)
        [0.0, 8.0],   # BL → (file=0, rank=0 bot) → board (0,8)
    ], dtype=np.float64)

    src = corners_np.astype(np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_point(H, px, py):
    """
    Apply homography H to a single point (px, py) in raw image space.
    Returns (bx, by) in 8x8 board space.
    """
    pt  = np.array([[[px, py]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    bx, by = float(out[0, 0, 0]), float(out[0, 0, 1])
    return bx, by


def center_to_square(bx, by):
    """
    Convert a point in 8x8 board space to a (row, col) cell and
    a python-chess square index.

    bx in [0, 8] → col (file)
    by in [0, 8] → row (0 = top = rank 8)

    Clamps to [0,7] to handle floating-point edge cases exactly on the
    board boundary.
    """
    col = int(min(7, max(0, int(bx))))  # file a-h = col 0-7
    row = int(min(7, max(0, int(by))))  # rank 8-1 = row 0-7

    rank   = 7 - row    # row 0 = rank 8 → rank_idx 7
    sq_idx = chess.square(col, rank)
    return sq_idx, row, col


def assign_pieces_to_squares(pieces, corners_np):
    """
    Full pipeline: compute homography, transform all piece centers,
    assign each to a square, resolve conflicts.

    Args:
        pieces:     list of dicts from piece_detector.detect_pieces()
                    each has "center": (cx, cy), "confidence": float,
                    "bbox": [xmin, ymin, xmax, ymax]
        corners_np: (4, 2) float32 corners [TL, TR, BR, BL]

    Returns:
        square_assignments: dict {sq_idx → piece_dict}
            Only squares with a detected piece are in this dict.
            Empty squares are absent (handled as Empty downstream).
            Each piece_dict has the original fields plus:
              "board_xy": (bx, by) in 8x8 space
              "sq_name":  e.g. "e4"

    Conflict resolution (Q5 decision):
        If multiple pieces map to the same square, keep the one with
        the highest detector confidence and discard the rest.
    """
    if not pieces or corners_np is None:
        return {}

    H = compute_homography(corners_np)

    # First pass — transform all centers, record square assignment
    candidates = []  # list of (sq_idx, piece_dict_extended)
    for piece in pieces:
        cx, cy = piece["center"]
        bx, by = transform_point(H, cx, cy)

        # Skip if center lands outside the board (bad detection artifact)
        if not (0.0 <= bx <= 8.0 and 0.0 <= by <= 8.0):
            continue

        sq_idx, row, col = center_to_square(bx, by)
        extended = dict(piece)
        extended["board_xy"] = (bx, by)
        extended["sq_name"]  = chess.square_name(sq_idx)
        candidates.append((sq_idx, extended))

    # Second pass — resolve conflicts (highest confidence wins per square)
    square_assignments = {}
    for sq_idx, piece_dict in candidates:
        if sq_idx not in square_assignments:
            square_assignments[sq_idx] = piece_dict
        else:
            existing = square_assignments[sq_idx]
            if piece_dict["confidence"] > existing["confidence"]:
                square_assignments[sq_idx] = piece_dict

    return square_assignments
