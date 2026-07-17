"""
Chess OCR — Classical CV Board Detection (Digital Pipeline Only)
====================================================================
Used ONLY for digital screenshots (Lichess, Chess.com, etc).
Physical boards now use the trained YOLOv8n-pose corner detector
(see corner_detector.py) — NOT Hough lines anymore (Section 21 pivot).

Strategy:
  1. Fast path — contour detection (works for clean screenshots)
  2. Robust fallback — alternating-square pattern scan (works when
     sidebars/coordinate-labels/UI chrome confuse contour detection)
  3. Colorfulness fallback — row/column "colorfulness" projection
     (works when strategies 1-2 fail on screenshots with dark UI
     chrome/borders around the board and/or highlighted squares,
     e.g. last-move/selection overlays. Neutral-gray chrome scores
     near-zero colorfulness regardless of brightness, so it survives
     the exact failure mode that breaks brightness-threshold contour
     detection: a dark border being merged into the board region.)
"""

import cv2
import numpy as np


# ── Public entrypoint ──────────────────────────────────────────────────────────

def detect_digital_board(img_np):
    """
    Detect chessboard in a digital screenshot.

    Returns: corners (4x2 float32 array, ordered [TL, TR, BR, BL])
             or None if all strategies fail
    """
    corners = _contour_detect(img_np)
    if corners is not None:
        return corners

    h, w    = img_np.shape[:2]
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    corners = _pattern_detect(gray, w, h)
    if corners is not None:
        return corners

    return _colorfulness_detect(img_np)


# ── Fast path: contour detection ──────────────────────────────────────────────

def _contour_detect(img_np):
    h, w = img_np.shape[:2]
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = np.ones((3, 3), np.uint8)
    edges   = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:15]:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05):
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = order_corners(corners)
            rect_w  = np.linalg.norm(corners[1] - corners[0])
            rect_h  = np.linalg.norm(corners[3] - corners[0])
            ratio   = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1)
            if ratio < 1.4:
                return corners

    return None


# ── Robust fallback: alternating square pattern scan ──────────────────────────

def _pattern_detect(gray, img_w, img_h):
    best_score   = 0
    best_corners = None

    for size_frac in [0.9, 0.8, 0.7, 0.6, 0.5]:
        board_size = int(min(img_w, img_h) * size_frac)
        sq_size    = board_size // 8
        if sq_size < 10:
            continue

        for x_off in [img_w // 2 - board_size // 2, int(img_w * 0.1), int(img_w * 0.05)]:
            for y_off in [img_h // 2 - board_size // 2, int(img_h * 0.1), int(img_h * 0.05)]:
                if x_off < 0 or y_off < 0:
                    continue
                if x_off + board_size > img_w or y_off + board_size > img_h:
                    continue

                score = _score_grid(gray, x_off, y_off, sq_size)
                if score > best_score:
                    best_score   = score
                    best_corners = np.array([
                        [x_off,              y_off             ],
                        [x_off + board_size, y_off             ],
                        [x_off + board_size, y_off + board_size],
                        [x_off,              y_off + board_size],
                    ], dtype=np.float32)

    return best_corners if best_score > 0.3 else None


def _score_grid(gray, x_off, y_off, sq_size):
    contrasts = []
    for row in range(8):
        for col in range(8):
            x1 = x_off + col * sq_size
            y1 = y_off + row * sq_size
            x2 = x1 + sq_size
            y2 = y1 + sq_size
            if y2 > gray.shape[0] or x2 > gray.shape[1]:
                return 0.0
            sq   = gray[y1:y2, x1:x2]
            mean = float(sq.mean())
            is_light = (row + col) % 2 == 0
            contrasts.append((mean, is_light))

    light_means = [m for m, light in contrasts if light]
    dark_means  = [m for m, light in contrasts if not light]
    if not light_means or not dark_means:
        return 0.0

    avg_light = sum(light_means) / len(light_means)
    avg_dark  = sum(dark_means)  / len(dark_means)
    return abs(avg_light - avg_dark) / 255.0


# ── Robust fallback #2: colorfulness projection ────────────────────────────────
#
# Contour detection and the brightness-based pattern scan both key off
# brightness/edges. That fails when the board sits inside dark UI chrome
# (a border/frame) whose brightness is similar to the board's dark squares —
# the contour merges chrome + board into one region, and the pattern scan's
# alternating-brightness score gets diluted by highlighted (non-alternating)
# squares. This strategy instead measures "colorfulness" — max(R,G,B) minus
# min(R,G,B) — per row/column. Neutral gray UI chrome is colorfulness ~0
# regardless of how dark or light it is; wood-tone/colored board squares are
# clearly colorful. Works even with highlighted (e.g. yellow/pink last-move)
# squares present, since those are still colorful, just a different hue.

def _colorfulness_detect(img_np, thresh=40.0):
    """
    Find the board's bounding box via row/column colorfulness projection.

    thresh: colorfulness cutoff separating neutral chrome (~0-10) from a
            colored board (~50-90 typical for wood/colored square themes).
            40.0 is a conservative midpoint tuned against real screenshots;
            revisit if this misfires on very muted/grayscale board skins.

    Returns corners (4x2 float32, [TL, TR, BR, BL]) or None if no clear
    high-colorfulness region is found.
    """
    if img_np.ndim != 3 or img_np.shape[2] < 3:
        return None

    arr = img_np[:, :, :3].astype(np.int16)
    colorfulness = arr.max(axis=2) - arr.min(axis=2)

    row_color = colorfulness.mean(axis=1)
    col_color = colorfulness.mean(axis=0)

    rows_board = np.where(row_color > thresh)[0]
    cols_board = np.where(col_color > thresh)[0]
    if len(rows_board) == 0 or len(cols_board) == 0:
        return None

    y1, y2 = int(rows_board.min()), int(rows_board.max())
    x1, x2 = int(cols_board.min()), int(cols_board.max())

    # sanity check: require a reasonably sized, non-degenerate region
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)


# ── Precise border-fit detector (clean vector-rendered diagrams) ──────────────
#
# Digital diagram renderers (as opposed to real screenshots/photos) typically
# draw a solid near-black outer frame around the board. Brightness-threshold
# Hough (grid_hough.detect_digital_board_hough) and Canny/approxPolyDP
# (_contour_detect above) both estimate that frame's corners only to the
# nearest detected line/polygon vertex, which is good enough for photos but
# leaves a few px of avoidable jitter on a perfectly straight, perfectly
# axis-aligned vector-drawn border -- jitter that naive equal-division
# slicing (slice_squares) then propagates uncorrected into every square
# crop (most visible as accumulating drift by row/col 7-8).
#
# This detector instead fits each of the 4 border sides with a subpixel
# least-squares line (cv2.fitLine) over ALL near-black border pixels along
# that side, rather than trusting a single polygon vertex. Intersecting the
# 4 fitted lines gives corner coordinates precise to a fraction of a pixel.
# Only fires when a solid near-black frame is actually present (cheap to
# check, returns None fast otherwise) -- e.g. it will not fire on photos,
# so it's safe to try first in the digital corner-detection cascade.

def _fit_line(side_pts):
    vx, vy, x0, y0 = cv2.fitLine(side_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def _intersect_lines(l1, l2):
    vx1, vy1, x1_, y1_ = l1
    vx2, vy2, x2_, y2_ = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    b = np.array([x2_ - x1_, y2_ - y1_])
    try:
        t, _ = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return (x1_ + t * vx1, y1_ + t * vy1)


def detect_digital_board_border(img_np, black_thresh=60, min_frac=0.5):
    """
    Find board corners via subpixel least-squares fit of a solid outer
    border, for vector-rendered diagram-style boards.

    Tries two strategies:
      1. Absolute near-black threshold (`_border_absolute`) -- fast, for
         genuinely black frames.
      2. Adaptive dark-fraction spike projection (`_border_adaptive`) --
         for wood-tone/colored border rules that aren't literally black
         (a fixed brightness cutoff either misses these entirely, or if
         raised far enough to catch them, starts re-merging with the
         checkerboard's own dark squares -- the same whole-canvas-blob
         failure mode this detector exists to avoid).

    Returns: corners (4x2 float32, [TL, TR, BR, BL]) or None if no plausible
             solid border is found by either strategy.
    """
    corners = _border_absolute(img_np, black_thresh=black_thresh, min_frac=min_frac)
    if corners is not None:
        return corners
    return _border_adaptive(img_np, min_frac=min_frac)


def _border_absolute(img_np, black_thresh=60, min_frac=0.5):
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    dark_mask = (gray < black_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    if bw < w * min_frac or bh < h * min_frac:
        return None
    ratio = max(bw, bh) / max(min(bw, bh), 1)
    if ratio > 1.25:
        return None

    pts = cnt.reshape(-1, 2).astype(np.float32)

    band = max(4, int(0.02 * min(bw, bh)))
    top_pts    = pts[pts[:, 1] < y + band]
    bottom_pts = pts[pts[:, 1] > y + bh - band]
    left_pts   = pts[pts[:, 0] < x + band]
    right_pts  = pts[pts[:, 0] > x + bw - band]

    if min(len(top_pts), len(bottom_pts), len(left_pts), len(right_pts)) < 10:
        return None  # not enough border pixels on one side for a reliable fit

    l_top, l_bottom = _fit_line(top_pts), _fit_line(bottom_pts)
    l_left, l_right = _fit_line(left_pts), _fit_line(right_pts)

    tl = _intersect_lines(l_top, l_left)
    tr = _intersect_lines(l_top, l_right)
    br = _intersect_lines(l_bottom, l_right)
    bl = _intersect_lines(l_bottom, l_left)

    if any(p is None for p in (tl, tr, br, bl)):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # sanity: fitted corners shouldn't stray far from the initial bbox
    # (guards against a degenerate near-parallel line-fit edge case)
    bbox_corners = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
    if np.max(np.linalg.norm(corners - bbox_corners, axis=1)) > 0.1 * min(bw, bh):
        return None

    return corners


def _border_adaptive(img_np, min_frac=0.5, search_frac=0.15, spike_thresh=0.75, strip=3):
    """
    Locate a solid border rule-line of *any* darkness (not just near-black)
    by scanning for a local spike in per-row/per-column dark-pixel fraction,
    relative to the image's own mean brightness.

    Digital diagram renderers commonly draw: [thin label margin] -> [solid
    border rule-line] -> [8x8 checkerboard]. The margin has a moderate,
    fairly flat dark-fraction (partial text coverage); the border rule-line
    is a sharp spike (near-100% dark clear across the full width/height);
    the checkerboard settles to ~50% (alternating light/dark squares). A
    single global brightness threshold can't separate "border" from
    "checkerboard" if their absolute darkness is similar -- but the spike
    in the *projection profile* is a distinct, scale-adaptive signature
    that a flat threshold can't see.

    Returns: corners (4x2 float32, [TL, TR, BR, BL]) or None if no clean
             spike is found on all four sides.
    """
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_b = gray.mean()
    bw_mask = (gray < mean_b).astype(np.uint8)

    row_frac = bw_mask.mean(axis=1)
    col_frac = bw_mask.mean(axis=0)

    band_h = max(15, int(search_frac * h))
    band_w = max(15, int(search_frac * w))

    def _spike_from_start(profile, band):
        idx = np.where(profile[:band] > spike_thresh)[0]
        if len(idx) == 0:
            return None
        i = int(idx[0])
        # confirm it's a thin rule-line, not entering a large filled block:
        # a bit further in, fraction should relax back down toward the
        # checkerboard's ~50% (or at least clearly off the spike).
        probe = min(len(profile) - 1, i + max(6, band // 4))
        if profile[probe] > spike_thresh - 0.1:
            return None
        return i

    def _spike_from_end(profile, band):
        rev = profile[::-1]
        i = _spike_from_start(rev, band)
        if i is None:
            return None
        return len(profile) - 1 - i

    top    = _spike_from_start(row_frac, band_h)
    bottom = _spike_from_end(row_frac, band_h)
    left   = _spike_from_start(col_frac, band_w)
    right  = _spike_from_end(col_frac, band_w)

    if any(v is None for v in (top, bottom, left, right)):
        return None
    if (right - left) < w * min_frac or (bottom - top) < h * min_frac:
        return None

    # Subpixel-refine each side with a line fit over dark pixels in a thin
    # strip straddling the detected spike row/column (mirrors _border_absolute).
    def _side_pts_row(row_idx):
        strip_mask = bw_mask[max(0, row_idx - strip):row_idx + strip + 1, :]
        ys, xs = np.where(strip_mask > 0)
        if len(xs) < 10:
            return None
        ys = ys + max(0, row_idx - strip)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _side_pts_col(col_idx):
        strip_mask = bw_mask[:, max(0, col_idx - strip):col_idx + strip + 1]
        ys, xs = np.where(strip_mask > 0)
        if len(xs) < 10:
            return None
        xs = xs + max(0, col_idx - strip)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    top_pts, bottom_pts = _side_pts_row(top), _side_pts_row(bottom)
    left_pts, right_pts = _side_pts_col(left), _side_pts_col(right)
    if any(p is None for p in (top_pts, bottom_pts, left_pts, right_pts)):
        # fall back to the coarse spike-index rectangle (still adaptive,
        # just not subpixel-refined)
        return np.array([[left, top], [right, top],
                          [right, bottom], [left, bottom]], dtype=np.float32)

    l_top, l_bottom = _fit_line(top_pts), _fit_line(bottom_pts)
    l_left, l_right = _fit_line(left_pts), _fit_line(right_pts)

    tl = _intersect_lines(l_top, l_left)
    tr = _intersect_lines(l_top, l_right)
    br = _intersect_lines(l_bottom, l_right)
    bl = _intersect_lines(l_bottom, l_left)

    if any(p is None for p in (tl, tr, br, bl)):
        return np.array([[left, top], [right, top],
                          [right, bottom], [left, bottom]], dtype=np.float32)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # guard against the fit straying far from the coarse spike estimate
    coarse = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    if np.max(np.linalg.norm(corners - coarse, axis=1)) > 0.1 * min(right - left, bottom - top):
        return coarse

    return corners


# ── Shared helper ──────────────────────────────────────────────────────────────

def order_corners(corners):
    """Order 4 corners as [top-left, top-right, bottom-right, bottom-left]."""
    corners    = sorted(corners, key=lambda p: p[1])
    top_two    = sorted(corners[:2], key=lambda p: p[0])
    bottom_two = sorted(corners[2:], key=lambda p: p[0])
    return np.array([top_two[0], top_two[1],
                     bottom_two[1], bottom_two[0]], dtype=np.float32)
