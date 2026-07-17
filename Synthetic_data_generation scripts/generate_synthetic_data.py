"""
Chess OCR — Synthetic Data Generator (Minimal)
================================================
Generates labelled 64x64 square crops from rendered chess positions.
Stops each class at MAX_PER_CLASS samples. No augmentation.

Output:
    organized_data/classification/synthetic/
        Empty/ wK/ wQ/ wR/ wB/ wN/ wP/ bK/ bQ/ bR/ bB/ bN/ bP/

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --out ./data/synthetic --max 50
"""

import os
import random
import argparse
import numpy as np
import chess
import chess.svg
import cairosvg
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from io import BytesIO
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = [
    "Empty",
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]

PIECE_TO_CLASS = {
    chess.Piece(chess.KING,   chess.WHITE): "wK",
    chess.Piece(chess.QUEEN,  chess.WHITE): "wQ",
    chess.Piece(chess.ROOK,   chess.WHITE): "wR",
    chess.Piece(chess.BISHOP, chess.WHITE): "wB",
    chess.Piece(chess.KNIGHT, chess.WHITE): "wN",
    chess.Piece(chess.PAWN,   chess.WHITE): "wP",
    chess.Piece(chess.KING,   chess.BLACK): "bK",
    chess.Piece(chess.QUEEN,  chess.BLACK): "bQ",
    chess.Piece(chess.ROOK,   chess.BLACK): "bR",
    chess.Piece(chess.BISHOP, chess.BLACK): "bB",
    chess.Piece(chess.KNIGHT, chess.BLACK): "bN",
    chess.Piece(chess.PAWN,   chess.BLACK): "bP",
}

LIGHT_SQUARE_COLOR = (200, 200, 200)
DARK_SQUARE_COLOR  = (100, 100, 100)
BOARD_RENDER_SIZE  = 512

# ── Board generators ──────────────────────────────────────────────────────────

def random_midgame_position():
    board = chess.Board()
    for _ in range(random.randint(5, 40)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

def dense_random_position():
    board = chess.Board(fen=None)
    all_pieces = [
        chess.Piece(pt, color)
        for pt in chess.PIECE_TYPES
        for color in [chess.WHITE, chess.BLACK]
    ]
    n_squares = int(64 * random.uniform(0.4, 0.65))
    for sq in random.sample(list(chess.SQUARES), n_squares):
        board.set_piece_at(sq, random.choice(all_pieces))
    return board

# ── Rendering ─────────────────────────────────────────────────────────────────

def render_board(board, size=BOARD_RENDER_SIZE):
    svg_str = chess.svg.board(board, size=size, coordinates=False)
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode())
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)

def slice_board(board_img, size=BOARD_RENDER_SIZE):
    sq_size = size // 8
    crops = {}
    for row in range(8):
        for col in range(8):
            crop = board_img.crop((
                col * sq_size, row * sq_size,
                (col + 1) * sq_size, (row + 1) * sq_size
            ))
            crop = crop.resize((64, 64), Image.LANCZOS)
            rank = 7 - row
            sq_idx = chess.square(col, rank)
            crops[sq_idx] = crop
    return crops

def get_label(board, square):
    piece = board.piece_at(square)
    return "Empty" if piece is None else PIECE_TO_CLASS[piece]

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop_img, square_index):
    file = chess.square_file(square_index)
    rank = chess.square_rank(square_index)
    is_light = (file + rank) % 2 == 1
    target_bg = float(LIGHT_SQUARE_COLOR[0] if is_light else DARK_SQUARE_COLOR[0])

    gray = np.array(crop_img.convert("L"), dtype=np.float32)
    corners = np.concatenate([
        gray[:4, :4].ravel(), gray[:4, -4:].ravel(),
        gray[-4:, :4].ravel(), gray[-4:, -4:].ravel(),
    ])
    shift = target_bg - corners.mean()
    gray = np.clip(gray + shift, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    return Image.fromarray(gray, mode="L").convert("RGB")

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset(out_dir="./../organized_data/classification/synthetic", max_per_class=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (out_path / cls).mkdir(exist_ok=True)

    class_counts = {cls: 0 for cls in CLASSES}
    position_idx = 0
    total_saved = 0

    print(f"Generating until all classes reach {max_per_class} samples...")
    print(f"Output: {out_path.resolve()}\n")

    while not all(class_counts[c] >= max_per_class for c in CLASSES):
        board = random_midgame_position() if random.random() < 0.6 else dense_random_position()
        board_img = render_board(board)
        crops = slice_board(board_img)

        for sq_idx, crop in crops.items():
            label = get_label(board, sq_idx)
            if class_counts[label] >= max_per_class:
                continue

            processed = preprocess_crop(crop, sq_idx)
            fname = f"{position_idx:06d}_sq{sq_idx:02d}.png"
            processed.save(out_path / label / fname)
            class_counts[label] += 1
            total_saved += 1

        position_idx += 1

        if position_idx % 50 == 0:
            done = sum(1 for c in CLASSES if class_counts[c] >= max_per_class)
            print(f"  Positions: {position_idx} | Images: {total_saved} | Classes full: {done}/{len(CLASSES)}")

    print(f"\nDONE — {total_saved} images saved across {len(CLASSES)} classes")
    print("\nPer-class counts:")
    for cls in CLASSES:
        print(f"  {cls:8s}: {class_counts[cls]}")
    print(f"\nDataset ready at: {out_path.resolve()}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./../organized_data/classification/synthetic")
    parser.add_argument("--max", type=int, default=50, dest="max_per_class")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(out_dir=args.out, max_per_class=args.max_per_class, seed=args.seed)
