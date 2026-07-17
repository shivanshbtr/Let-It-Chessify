"""
Chess Piece Downloader, Extractor & Augmentor
===============================================
Single script that:
  1. Downloads chess piece sets from chess.com CDN (34 normal themes × 12 pieces)
  2. Applies the same preprocessing pipeline as generate_synthetic_data.py
     (grayscale → bg normalize → CLAHE)
  3. Augments each piece 9-10 times with the same augment_crop() logic
  4. Saves everything into data/classification/synthetic/<label>/

Special handling — blindfold theme:
  blindfold pieces are blank squares (no visible piece) so ALL 12 piece
  PNGs from that theme are routed to Empty/ instead of their piece labels.
  This gives high-quality, diverse Empty square samples across light/dark bgs.

Label folders match train_classifier.py LABEL_MAP exactly:
    data/classification/synthetic/
        Empty/
        wK/ wQ/ wR/ wB/ wN/ wP/
        bK/ bQ/ bR/ bB/ bN/ bP/

Usage:
    python download_pieces.py
    python download_pieces.py --out ./data/classification/synthetic --augments 10
    python download_pieces.py --themes alpha neo wood --augments 9
    python download_pieces.py --list-themes
    python download_pieces.py --dry-run

Install deps:
    pip install httpx Pillow numpy opencv-python
"""

import argparse
import io
import random
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ── CDN ───────────────────────────────────────────────────────────────────────

CDN_URL      = "https://images.chesscomfiles.com/chess-themes/pieces/{theme}/{size}/{piece}.png"
DEFAULT_SIZE = 150

# ── Themes ────────────────────────────────────────────────────────────────────

# blindfold is handled separately — its pieces are blank squares → Empty/
BLINDFOLD_THEME = "blindfold"

# Verified from images.chesscomfiles.com CDN (default-config.json)
# midnight and minimalist removed — they don't exist on the CDN
# Added: 3d_chesskid, 3d_plastic, 3d_staunton, 3d_wood, 8_bit, newspaper
ALL_THEMES = [
    "alpha", "neo", "neo_wood", "wood", "bases", "book",
    "bubblegum", "cases", "classic", "club", "condal", "dash", "game_room",
    "glass", "gothic", "graffiti", "icy_sea", "light", "lolz", "marble",
    "maya", "metal", "modern", "nature", "neon", "newspaper",
    "ocean", "sky", "space", "tigers", "tournament", "vintage",
    "3d_chesskid", "3d_plastic", "3d_staunton", "3d_wood", "8_bit",
]

# ── Piece map ─────────────────────────────────────────────────────────────────

PIECE_CDN_NAMES = ["wp", "wk", "wq", "wr", "wb", "wn",
                   "bp", "bk", "bq", "br", "bb", "bn"]

FILENAME_TO_LABEL = {
    "wp": "wP", "wk": "wK", "wq": "wQ", "wr": "wR", "wb": "wB", "wn": "wN",
    "bp": "bP", "bk": "bK", "bq": "bQ", "br": "bR", "bb": "bB", "bn": "bN",
}

ALL_LABELS = ["Empty"] + list(FILENAME_TO_LABEL.values())

LIGHT_SQUARE_COLOR = 200.0
DARK_SQUARE_COLOR  = 100.0

# ── Terminal colours ──────────────────────────────────────────────────────────

def bold(s):   return f"\033[1m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"

# ── Preprocessing (identical to generate_synthetic_data.py) ──────────────────

def composite_on_bg(piece_img: Image.Image, bg_color: int, size: int) -> Image.Image:
    bg = Image.new("RGBA", piece_img.size, (bg_color, bg_color, bg_color, 255))
    composited = Image.alpha_composite(bg, piece_img.convert("RGBA")).convert("L")
    return composited.resize((size, size), Image.LANCZOS)


def normalize_square_color(gray_img: Image.Image, is_light: bool) -> Image.Image:
    target_bg = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
    gray      = np.array(gray_img, dtype=np.float32)
    corners   = np.concatenate([
        gray[:4,  :4 ].ravel(), gray[:4,  -4:].ravel(),
        gray[-4:, :4 ].ravel(), gray[-4:, -4:].ravel(),
    ])
    shift = target_bg - corners.mean()
    return Image.fromarray(np.clip(gray + shift, 0, 255).astype(np.uint8), mode="L")


def apply_clahe(gray_img: Image.Image) -> Image.Image:
    arr   = np.array(gray_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return Image.fromarray(clahe.apply(arr), mode="L")


def preprocess(piece_img: Image.Image, is_light: bool, out_size: int = 64) -> Image.Image:
    bg_color = int(LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR)
    gray     = composite_on_bg(piece_img, bg_color, out_size)
    gray     = normalize_square_color(gray, is_light)
    gray     = apply_clahe(gray)
    return gray.convert("RGB")


# ── Augmentation (identical to generate_synthetic_data.py) ───────────────────

def augment_crop(img: Image.Image) -> Image.Image:
    img = img.copy()
    if random.random() > 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.4:
        img = img.rotate(random.uniform(-8, 8), fillcolor=(128, 128, 128))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    if random.random() > 0.5:
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
        img   = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img


# ── Download ──────────────────────────────────────────────────────────────────

def fetch(url: str, retries: int = 3, wait_ms: int = 200) -> bytes | None:
    time.sleep(wait_ms / 1000)
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(url, timeout=15, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt < retries:
                time.sleep(attempt * 0.5)
    return None


# ── Save piece with augments ──────────────────────────────────────────────────

def process_piece(
    piece_img:  Image.Image,
    label:      str,           # destination label folder e.g. "wP" or "Empty"
    theme:      str,
    piece_name: str,           # cdn name e.g. "wp"
    out_dir:    Path,
    augments:   int,
) -> int:
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for is_light, sq_tag in [(True, "light"), (False, "dark")]:
        base      = preprocess(piece_img, is_light)
        base_name = f"{theme}_{piece_name}_{sq_tag}_base.png"
        base.save(label_dir / base_name)
        saved += 1

        n_aug = max(1, augments // 2)
        for i in range(n_aug):
            aug = augment_crop(base)
            aug.save(label_dir / f"{theme}_{piece_name}_{sq_tag}_aug{i:02d}.png")
            saved += 1

    return saved


# ── Blindfold → Empty ─────────────────────────────────────────────────────────

def process_blindfold(out_dir: Path, cdn_size: int, augments: int, dry_run: bool) -> int:
    """
    Download all 12 blindfold piece PNGs and save every one to Empty/
    since blindfold squares show no piece — just the bare square color.
    Each PNG gives us one more bg variety for Empty samples.
    """
    print(bold(f"\n  [blindfold] → Empty/  (blank squares, no piece visible)"))

    empty_dir = out_dir / "Empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    total_fail  = 0

    for piece_name in PIECE_CDN_NAMES:
        url        = CDN_URL.format(theme=BLINDFOLD_THEME, size=cdn_size, piece=piece_name)
        base_light = empty_dir / f"blindfold_{piece_name}_light_base.png"

        if base_light.exists():
            n = len(list(empty_dir.glob(f"blindfold_{piece_name}_*.png")))
            print(f"    {yellow('skip')}  blindfold/{piece_name}.png → Empty/ ({n} files exist)")
            total_saved += n
            continue

        if dry_run:
            print(f"    {bold('dry')}   {url}  → Empty/")
            continue

        img_bytes = fetch(url)
        if img_bytes is None:
            print(f"    {red('fail')}  blindfold/{piece_name}.png — download failed")
            total_fail += 1
            continue

        piece_img = Image.open(io.BytesIO(img_bytes))
        n_saved   = process_piece(piece_img, "Empty", BLINDFOLD_THEME,
                                  piece_name, out_dir, augments)

        print(f"    {green('ok')}    blindfold/{piece_name}.png → Empty/  ({n_saved} images)")
        total_saved += n_saved

    status = green(f"✓ {total_saved} Empty images from blindfold")
    if total_fail:
        status += red(f"  {total_fail} failed")
    print(f"  └─ {status}\n")
    return total_saved


# ── Core ──────────────────────────────────────────────────────────────────────

def download_and_augment(
    themes:   list[str],
    out_dir:  Path,
    cdn_size: int,
    augments: int,
    dry_run:  bool,
):
    for label in ALL_LABELS:
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    imgs_per_piece = 2 + augments
    total_expected = len(themes) * len(PIECE_CDN_NAMES) * imgs_per_piece
    # blindfold adds 12 piece PNGs × imgs_per_piece all going to Empty
    blindfold_expected = len(PIECE_CDN_NAMES) * imgs_per_piece

    print(bold(f"\n{'═'*58}"))
    print(bold("  Chess Piece Downloader + Augmentor"))
    print(f"{'═'*58}")
    print(f"  Normal themes   : {len(themes)}  ({len(themes) * len(PIECE_CDN_NAMES)} pieces)")
    print(f"  Blindfold theme : 12 PNGs → all go to Empty/")
    print(f"  Augments/piece  : {augments}  (+ 2 bases = {imgs_per_piece} per piece)")
    print(f"  Expected total  : ~{total_expected + blindfold_expected} images")
    print(f"  Output          : {out_dir.resolve()}")
    print(f"{'═'*58}\n")

    total_saved = 0
    total_fail  = 0
    failed      = []

    # ── 1. Normal themes → respective label folders ───────────────────────────
    for theme in themes:
        theme_saved = 0
        theme_fail  = 0

        print(bold(f"  [{theme}]"))

        for piece_name in PIECE_CDN_NAMES:
            label      = FILENAME_TO_LABEL[piece_name]
            url        = CDN_URL.format(theme=theme, size=cdn_size, piece=piece_name)
            base_light = out_dir / label / f"{theme}_{piece_name}_light_base.png"

            if base_light.exists():
                n = len(list((out_dir / label).glob(f"{theme}_{piece_name}_*.png")))
                print(f"    {yellow('skip')}  {piece_name} → {label}/ ({n} files exist)")
                theme_saved += n
                continue

            if dry_run:
                print(f"    {bold('dry')}   {url}")
                continue

            img_bytes = fetch(url)
            if img_bytes is None:
                print(f"    {red('fail')}  {piece_name} — download failed")
                theme_fail += 1
                failed.append(url)
                continue

            piece_img = Image.open(io.BytesIO(img_bytes))
            n_saved   = process_piece(piece_img, label, theme, piece_name,
                                      out_dir, augments)

            print(f"    {green('ok')}    {piece_name} → {label}/  "
                  f"({n_saved} images: 2 base + {augments} aug)")
            theme_saved += n_saved

        total_saved += theme_saved
        total_fail  += theme_fail
        status = green(f"✓ {theme_saved} images")
        if theme_fail:
            status += red(f"  {theme_fail} failed")
        print(f"  └─ {status}\n")

    # ── 2. Blindfold → Empty/ ─────────────────────────────────────────────────
    empty_saved  = process_blindfold(out_dir, cdn_size, augments, dry_run)
    total_saved += empty_saved

    # ── Summary ───────────────────────────────────────────────────────────────
    print(bold(f"{'═'*58}"))
    print(bold("  Summary"))
    print(f"{'═'*58}")
    print(f"  Total images saved : {total_saved}")
    print(f"  Failed downloads   : {total_fail}")
    print(f"\n  Label folder counts:")
    for label in ALL_LABELS:
        count = len(list((out_dir / label).glob("*.png")))
        bar   = "█" * min(count // 10, 30)
        print(f"    {'Empty' if label == 'Empty' else label:<6} → {count:>4} images  {bar}")

    if failed:
        print(f"\n  {red('Failed URLs:')}")
        for u in failed:
            print(f"    {u}")

    print(f"\n  Next step: python build_splits.py")
    print(f"{'═'*58}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download chess.com pieces → preprocess + augment → organized_data/classification/synthetic/"
    )
    p.add_argument("--out",         default="./../organized_data/classification/synthetic",
                   help="Output root (default: ./../organized_data/classification/synthetic)")
    p.add_argument("--cdn-size",    type=int, default=DEFAULT_SIZE,
                   help=f"Piece size to request from CDN in px (default: {DEFAULT_SIZE})")
    p.add_argument("--augments",    type=int, default=9,
                   help="Augmented copies per piece per square color (default: 9)")
    p.add_argument("--themes",      nargs="+", default=None,
                   help="Specific themes to download (default: all, blindfold always included)")
    p.add_argument("--list-themes", action="store_true",
                   help="Print all available themes and exit")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print what would be downloaded without doing it")
    args = p.parse_args()

    if args.list_themes:
        print(f"\nNormal themes ({len(ALL_THEMES)}) — saved to their piece label folders:")
        for t in ALL_THEMES:
            print(f"  {t}")
        print(f"\nSpecial theme (always included):")
        print(f"  blindfold  → all pieces saved to Empty/")
        print()
        exit(0)

    themes  = args.themes if args.themes else ALL_THEMES
    # always strip blindfold from normal themes in case user passed it — handled separately
    themes  = [t for t in themes if t != BLINDFOLD_THEME]
    unknown = [t for t in themes if t not in ALL_THEMES]
    if unknown:
        print(yellow(f"Warning: unknown theme(s) {unknown} — will try anyway"))

    download_and_augment(
        themes   = themes,
        out_dir  = Path(args.out),
        cdn_size = args.cdn_size,
        augments = args.augments,
        dry_run  = args.dry_run,
    )
