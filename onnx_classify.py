"""
Chess Square Classifier — ONNX Inference
==========================================
For raw screenshot crops (chess.com, Lichess, any board theme).
Applies the same preprocessing pipeline as generate_synthetic_data.py
before feeding into the ONNX model.

Use --save-preprocessed to save the preprocessed image locally for visual inspection.

Usage:
    python onnx_classify.py --model square_classifier.onnx --image square.png --square e4
    python onnx_classify.py --model square_classifier.onnx --image square.png --square e4 --save-preprocessed
    python onnx_classify.py --model square_classifier.onnx --image square.png --square e4 --save-preprocessed --preview-dir ./debug

Install deps:
    pip install onnxruntime Pillow numpy opencv-python
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import onnxruntime as ort

# ── Label map ────────────────────────────────────────────────────────────────

DEFAULT_LABEL_MAP = {
    "Empty": 0,
    "wK": 1, "wQ": 2, "wR": 3, "wB": 4, "wN": 5, "wP": 6,
    "bK": 7, "bQ": 8, "bR": 9, "bB": 10, "bN": 11, "bP": 12,
}

PIECE_DISPLAY = {
    "Empty": "· Empty",
    "wK": "♔ White King",   "wQ": "♕ White Queen",  "wR": "♖ White Rook",
    "wB": "♗ White Bishop", "wN": "♘ White Knight",  "wP": "♙ White Pawn",
    "bK": "♚ Black King",   "bQ": "♛ Black Queen",   "bR": "♜ Black Rook",
    "bB": "♝ Black Bishop", "bN": "♞ Black Knight",  "bP": "♟ Black Pawn",
}

LIGHT_SQUARE_COLOR = 200.0
DARK_SQUARE_COLOR  = 100.0

# ── Preprocessing ─────────────────────────────────────────────────────────────

def is_light_square(square_name: str) -> bool:
    file = ord(square_name[0].lower()) - ord('a')
    rank = int(square_name[1]) - 1
    return (file + rank) % 2 == 1


def normalize_square_color(crop_img: Image.Image, is_light: bool) -> Image.Image:
    target_bg = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
    gray = np.array(crop_img.convert("L"), dtype=np.float32)
    corners = np.concatenate([
        gray[:4,  :4 ].ravel(),
        gray[:4,  -4:].ravel(),
        gray[-4:, :4 ].ravel(),
        gray[-4:, -4:].ravel(),
    ])
    current_bg = corners.mean()
    shift = target_bg - current_bg
    gray_shifted = np.clip(gray + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(gray_shifted, mode="L")


def apply_clahe(crop_img: Image.Image) -> Image.Image:
    img_np = np.array(crop_img)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_np = clahe.apply(img_np)
    return Image.fromarray(img_np, mode="L")


def preprocess(image_path: str, is_light: bool, save_dir: str = None, square_name: str = None):
    """
    Full pipeline matching preprocess_crop() + make_transforms(train=False).
    If save_dir is given, saves intermediate + final images there for inspection.

    Saved files:
      <square>_0_original.png       — raw input resized to 64x64
      <square>_1_grayscale.png      — after convert("L")
      <square>_2_bg_normalized.png  — after background shift
      <square>_3_clahe.png          — after CLAHE
      <square>_4_final_input.png    — what the model actually sees (denormalized back to [0,255])
    """
    img = Image.open(image_path).convert("RGB")

    # ── Step 0: original resized ──
    original_64 = img.resize((64, 64), Image.LANCZOS)

    # ── Step 1: grayscale ──
    gray = img.convert("L")

    # ── Step 2: background normalization ──
    bg_normalized = normalize_square_color(img, is_light)

    # ── Step 3: CLAHE ──
    clahe_img = apply_clahe(bg_normalized)

    # ── Step 4: resize + normalize ──
    resized = clahe_img.resize((64, 64), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5   # [-1, 1]

    # ── Save intermediates if requested ──
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = square_name.lower() if square_name else Path(image_path).stem

        original_64.save(out / f"{prefix}_0_original.png")

        gray.resize((64, 64), Image.LANCZOS).save(out / f"{prefix}_1_grayscale.png")

        bg_normalized.resize((64, 64), Image.LANCZOS).save(out / f"{prefix}_2_bg_normalized.png")

        clahe_img.resize((64, 64), Image.LANCZOS).save(out / f"{prefix}_3_clahe.png")

        # Denormalize [-1,1] back to [0,255] so it's viewable
        final_viewable = ((arr * 0.5 + 0.5) * 255).astype(np.uint8)
        Image.fromarray(final_viewable, mode="L").save(out / f"{prefix}_4_final_input.png")

        print(f"\n  Preprocessed images saved to: {out.resolve()}/")
        print(f"    {prefix}_0_original.png       ← raw input")
        print(f"    {prefix}_1_grayscale.png      ← after grayscale")
        print(f"    {prefix}_2_bg_normalized.png  ← after bg shift (target: {int(LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR)})")
        print(f"    {prefix}_3_clahe.png          ← after CLAHE")
        print(f"    {prefix}_4_final_input.png    ← what the model sees")

    return arr[np.newaxis, np.newaxis, :, :]   # [1, 1, 64, 64]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ── Inference ─────────────────────────────────────────────────────────────────

def classify(model_path: str, image_path: str, square_name: str,
             label_map: dict, top_k: int = 5,
             save_preprocessed: bool = False, preview_dir: str = "./debug"):

    idx_to_label = {v: k for k, v in label_map.items()}
    light = is_light_square(square_name)

    # Use CUDA if available, otherwise CPU — no warning either way
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    tensor = preprocess(
        image_path, light,
        save_dir=preview_dir if save_preprocessed else None,
        square_name=square_name,
    )

    outputs = session.run(["logits"], {"square_crop": tensor})
    logits  = outputs[0][0]
    probs   = softmax(logits)
    top_idx = np.argsort(probs)[::-1][:top_k]

    sq_type = "light" if light else "dark"
    print(f"\n  Image  : {image_path}  (square {square_name.lower()} — {sq_type})")
    print(f"{'─'*44}")
    print(f"  {'#':<4} {'Prob':>7}   {'Class':<10} Piece")
    print(f"{'─'*44}")
    for rank, idx in enumerate(top_idx, 1):
        label   = idx_to_label.get(idx, f"class_{idx}")
        display = PIECE_DISPLAY.get(label, label)
        marker  = " ◀" if rank == 1 else ""
        print(f"  #{rank:<3} {probs[idx]*100:>6.2f}%   {label:<10} {display}{marker}")
    print(f"{'─'*44}")

    best_label = idx_to_label.get(top_idx[0], f"class_{top_idx[0]}")
    print(f"\n  → {PIECE_DISPLAY.get(best_label, best_label)}  ({probs[top_idx[0]]*100:.2f}% confidence)\n")

    return best_label, float(probs[top_idx[0]])

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run chess square ONNX classifier on raw screenshot crops.")
    p.add_argument("--model",              required=True,        help="Path to square_classifier.onnx")
    p.add_argument("--image",              required=True,        help="Path to a raw square crop image")
    p.add_argument("--square",             default="a1",         help="Board square e.g. e4, a1, h8 (default: a1)")
    p.add_argument("--label-map",          default=None,         help="Path to label_map.json (optional)")
    p.add_argument("--top",                type=int, default=5,  help="Top-N predictions to show")
    p.add_argument("--save-preprocessed", action="store_true",   help="Save intermediate preprocessing steps locally")
    p.add_argument("--preview-dir",        default="./debug",    help="Where to save preprocessed images (default: ./debug)")
    args = p.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    label_map = DEFAULT_LABEL_MAP
    if args.label_map:
        with open(args.label_map) as f:
            label_map = json.load(f)

    classify(
        args.model, args.image, args.square, label_map, args.top,
        save_preprocessed=args.save_preprocessed,
        preview_dir=args.preview_dir,
    )
