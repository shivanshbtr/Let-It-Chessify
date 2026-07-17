"""
Chessboard grid detection + 64-square segmentation
====================================================

Generalizable pipeline for extracting 64 individual square crops from a photo
of a chessboard (phone photo of a screen, real board, etc.), even when the
shot is not perfectly perpendicular (trapezoidal / keystoned perspective).

WHY THIS APPROACH (not cv2.findChessboardCorners):
  cv2.findChessboardCorners / findChessboardCornersSB are built for camera
  CALIBRATION patterns -- they need a fully empty, high-contrast black/white
  board with all inner corners unobstructed. A real chess position has
  pieces sitting on top of the board occluding many of those corners, and
  many board themes have very low light/dark square contrast. Both make the
  built-in OpenCV chessboard-corner detectors fail (confirmed on the test
  image: both return found=False). So instead we detect the board's grid
  *lines* (robust to partial occlusion by pieces) rather than requiring
  every inner corner to be individually visible.

PIPELINE
  1. CLAHE (adaptive local contrast) to bring out subtle square-to-square
     contrast that global thresholding misses.
  2. Canny edge detection.
  3. Standard Hough transform (rho, theta form) -- unlike the probabilistic
     variant this naturally represents *slanted/converging* lines, which is
     exactly what a trapezoidal photo produces, so no assumption of
     perfectly horizontal/vertical lines is required.
  4. Split detected lines into two orientation families (the board's two
     principal axes) and cluster near-duplicate lines together.
  5. Within each family, robustly fit a 9-line evenly-spaced progression
     (RANSAC-style search over spacing + offset) so that spurious lines
     (piece edges, outer border/mat, glare) are ignored and any *missing*
     grid line is still recovered by interpolation.
  6. Intersect every vertical line with every horizontal line -> a 9x9 grid
     of (x, y) corner points. Because each line keeps its own (rho, theta),
     this grid correctly follows a trapezoidal/keystoned board shape -- it
     is NOT assumed to be an axis-aligned rectangle.
  7. For each of the 64 cells, take its own 4 corner points from the 9x9
     grid and warp *that single cell* to a fixed-size square with its own
     perspective transform. Doing this per-cell (rather than one global
     homography for the whole board) is the key trick that makes the method
     robust to uneven/trapezoidal perspective: every cell gets locally
     corrected independently.
  8. Save the 64 squares (row-major, row 0 = top of image, col 0 = left of
     image, in *image* orientation -- this script does not know which side
     is White, since that depends on camera orientation, so relabel to
     a1..h8 afterwards once you know board orientation) plus a contact-sheet
     mosaic and a debug overlay for visual QA.

USAGE
    python chessboard_grid_segmentation.py path/to/board.jpg [output_dir]

LIMITATIONS / WHEN TO ADD A MANUAL FALLBACK
  This line-based method needs the 8x8 grid pattern to be at least weakly
  visible (some pixel-level contrast between adjacent squares along most of
  the board). If a photo has extreme glare, motion blur, or a board skin
  with near-zero contrast AND heavy piece occlusion, automatic detection can
  fail. For a production tool, wrap this with a fallback UI that lets a user
  drag the 4 outer corners by hand when `find_board_grid()` raises
  RuntimeError -- the rest of the pipeline (steps 6-8) works unchanged once
  you have 4 correct outer corners (just linearly interpolate the 9x9 grid
  from those 4 corners instead of the Hough-detected lines).
"""

import sys
import os
import numpy as np
import cv2


def _normalize_line(rho, theta):
    """Bring theta into [0, pi), flipping rho's sign to match."""
    if theta < 0:
        theta += np.pi
        rho = -rho
    if theta >= np.pi:
        theta -= np.pi
    return rho, theta


def _cluster_1d(vals, gap):
    """Merge near-duplicate scalar values (within `gap`) into their mean."""
    vals = sorted(vals)
    clusters, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return [float(np.mean(c)) for c in clusters]


def _best_n_line_progression(rhos, n=9):
    """
    Robustly find the best evenly-spaced progression of n lines among noisy
    candidate rho values (RANSAC-style: try candidate spacings/anchors,
    keep the hypothesis explaining the most candidates with least error,
    then least-squares refine).

    Returns (fitted_positions[n], spacing) or None if no good fit exists.
    """
    rhos = sorted(rhos)
    if len(rhos) < 2:
        return None
    best = None
    spacing_candidates = set()
    for i in range(len(rhos)):
        for j in range(i + 1, len(rhos)):
            diff = rhos[j] - rhos[i]
            for k in range(1, n):
                s = round(diff / k, 1)
                if s > 3:
                    spacing_candidates.add(s)

    for spacing in spacing_candidates:
        for anchor in rhos:
            for start_offset in range(-n + 1, 1):
                expected = [anchor + (start_offset + k) * spacing for k in range(n)]
                matched, total_err, ok = [], 0.0, True
                for e in expected:
                    dists = [abs(e - r) for r in rhos]
                    idx = int(np.argmin(dists))
                    if dists[idx] > spacing * 0.35:
                        ok = False
                        break
                    matched.append(rhos[idx])
                    total_err += dists[idx]
                if not ok:
                    continue
                score = (len(set(matched)), -total_err)
                if best is None or score > best[0]:
                    ks = np.arange(n)
                    A = np.vstack([np.ones(n), ks]).T
                    sol, *_ = np.linalg.lstsq(A, np.array(matched), rcond=None)
                    a0, d0 = sol
                    fitted = [a0 + d0 * k for k in ks]
                    best = (score, fitted, spacing)
    if best is None:
        return None
    return best[1], best[2]


def _line_intersection(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None  # near-parallel lines (can happen with noisy family assignment) -- skip
    return np.linalg.solve(A, b)  # (x, y)


def _canonicalize_vertical_family(family_v):
    """
    A near-vertical Hough line can be numerically represented with theta
    near 0 OR near pi -- both pass the "closer to 0/pi than to pi/2" test
    used to sort lines into the vertical family, and both represent the
    same physical line orientation. But rho = x*cos(theta) + y*sin(theta):
    for theta near 0, cos(theta)~+1, so rho tracks +x (small rho = left
    side); for theta near pi, cos(theta)~-1, so rho tracks -x (small rho =
    RIGHT side). Left unfixed, whichever representation cv2.HoughLines
    happens to return for a given image silently flips which side of the
    image col=0 lands on in the final grid, independent of anything about
    the image's actual content -- this is not a hypothetical: confirmed
    empirically, 2 of 8 real test images came out horizontally mirrored
    (columns numbered right-to-left) purely from this ambiguity, with no
    correlation to image content.

    Fix: (rho, theta) and (-rho, theta - pi) describe the exact same
    infinite line (substitute into cos(theta)x + sin(theta)y = rho and
    the equation is identical after the sign flip -- cos(theta-pi) =
    -cos(theta), sin(theta-pi) = -sin(theta)), so remapping any near-pi
    representation to its near-0 equivalent changes nothing about the
    line's geometry or any downstream intersection computation, but makes
    every vertical line's rho consistently track true x-position before
    clustering/sorting.
    """
    out = []
    for rho, theta in family_v:
        if theta > np.pi / 2:
            rho, theta = -rho, theta - np.pi
        out.append((rho, theta))
    return out


def _cluster_lines_2d(family, gap):
    """
    Cluster (rho, theta) lines by their rho proximity, but keep each
    cluster's *actual mean (rho, theta)* -- unlike _cluster_1d this does not
    discard theta, which matters for converging (trapezoidal) lines where
    different grid lines in the same family have measurably different
    angles.
    """
    family = sorted(family, key=lambda rt: rt[0])
    clusters, cur = [], [family[0]]
    for rt in family[1:]:
        if rt[0] - cur[-1][0] <= gap:
            cur.append(rt)
        else:
            clusters.append(cur)
            cur = [rt]
    clusters.append(cur)
    return [(float(np.mean([r for r, t in c])), float(np.mean([t for r, t in c]))) for c in clusters]


def _autocrop_borders(img, mad_thresh=10.0, max_crop_frac=0.30, iterations=3):
    """
    Trim off solid/near-uniform-color margins from each side of the image
    before grid detection (e.g. app-UI letterboxing: black banners, side
    panels, background mats around the actual board). These are extremely
    common in phone-screenshot-of-an-app photos and are dangerous for the
    line-detection stage because a single long, high-contrast bar edge
    generates a much stronger Hough vote than the subtle internal square
    boundaries, and can hijack the grid-line fit.

    A row/column is considered "border" only if it is genuinely flat: median
    absolute deviation (MAD) from its own median < mad_thresh. MAD (not std)
    is deliberate: on a real phone photo, sensor grain and JPEG blocking
    noise inflate plain standard deviation on an otherwise solid-color bar
    enough that it stops reading as "flat" at all (observed on a real photo
    test case: a solid black title bar had std ~45 purely from noise, while
    its MAD was ~5) -- std is not robust to that kind of noise because a
    few outlier pixels move it a lot, whereas MAD (built on the median) is
    barely affected by a scattering of noisy pixels. An earlier version
    also treated "mean brightness close to the image corner color" as
    border evidence, but on a low-contrast board (adjacent squares only
    ~10-20 gray levels apart) that criterion kept re-triggering on real
    board content as the crop shrank and the corner reference drifted
    inward, eating further into the board each iteration -- MAD doesn't
    have that failure mode, and the caller additionally races this result
    against the uncropped original and scores both (see
    find_board_grid_robust), so an overly aggressive crop here is not fatal.

    Runs several row-crop/col-crop passes rather than a single pass: many
    UI layouts have banners that only span part of the width (e.g. a top
    banner with light margins on either side, framed by separate full-height
    side panels), so trimming the sides first, then re-measuring top/bottom
    on the now-narrower image, is needed before the banner reads as
    uniformly flat across the whole row. Each pass is capped at
    max_crop_frac of that dimension as a safety limit.

    Returns (cropped_img, x_offset, y_offset).
    """
    x_off, y_off = 0, 0
    cur = img
    for _ in range(iterations):
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape

        row_med = np.median(gray, axis=1, keepdims=True)
        row_mad = np.median(np.abs(gray - row_med), axis=1)
        col_med = np.median(gray, axis=0, keepdims=True)
        col_mad = np.median(np.abs(gray - col_med), axis=0)

        top = 0
        max_top = int(H * max_crop_frac)
        while top < max_top and row_mad[top] < mad_thresh:
            top += 1

        bottom = H - 1
        max_bottom = H - int(H * max_crop_frac)
        while bottom > max_bottom and row_mad[bottom] < mad_thresh:
            bottom -= 1

        left = 0
        max_left = int(W * max_crop_frac)
        while left < max_left and col_mad[left] < mad_thresh:
            left += 1

        right = W - 1
        max_right = W - int(W * max_crop_frac)
        while right > max_right and col_mad[right] < mad_thresh:
            right -= 1

        if top == 0 and left == 0 and bottom == H - 1 and right == W - 1:
            break  # stable, nothing more to trim

        cur = cur[top:bottom + 1, left:right + 1]
        x_off += left
        y_off += top

    return cur, x_off, y_off


def _find_grid_candidates(img, n=9, hough_thresholds=(140, 120, 100, 80, 60)):
    """
    Core detector: returns a list of (score, grid, info) candidates sorted
    best-first, or [] if nothing was found. See find_board_grid_robust for
    the public entry point and full explanation of the approach.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = cv2.GaussianBlur(eq, (3, 3), 0)
    edges = cv2.Canny(eq, 40, 120)

    last_err = None
    candidates = []  # (score, grid, debug_info)

    for thresh in hough_thresholds:
        lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=thresh)
        if lines is None or len(lines) < 2 * n:
            last_err = f"only found {0 if lines is None else len(lines)} Hough lines at threshold {thresh}"
            continue
        lines = lines[:, 0, :]
        norm = [_normalize_line(r, t) for r, t in lines]

        family_v, family_h = [], []
        for rho, theta in norm:
            d0 = min(theta, np.pi - theta)
            d90 = abs(theta - np.pi / 2)
            (family_v if d0 < d90 else family_h).append((rho, theta))

        family_v = _canonicalize_vertical_family(family_v)  # fix rho-sign ambiguity (see docstring)

        if len(family_v) < n or len(family_h) < n:
            last_err = f"unbalanced line families at threshold {thresh} (v={len(family_v)}, h={len(family_h)})"
            continue

        clusters_v = _cluster_lines_2d(family_v, gap=12)  # [(rho,theta), ...]
        clusters_h = _cluster_lines_2d(family_h, gap=12)

        # Use the (rho-only) progression search just to find the observed
        # candidate -> grid-index correspondence, not to extrapolate values.
        fit_v = _best_n_line_progression([r for r, t in clusters_v], n=n)
        fit_h = _best_n_line_progression([r for r, t in clusters_h], n=n)
        if fit_v is None or fit_h is None:
            last_err = f"could not fit {n}-line index correspondence at threshold {thresh}"
            continue

        fitted_v, spacing_v = fit_v
        fitted_h, spacing_h = fit_h

        def index_map(clusters, fitted_positions, spacing):
            """map grid index -> actual observed (rho,theta), or None if unmatched"""
            out = {}
            for idx, expected in enumerate(fitted_positions):
                best_c, best_d = None, None
                for (r, t) in clusters:
                    d = abs(r - expected)
                    if best_d is None or d < best_d:
                        best_d, best_c = d, (r, t)
                out[idx] = best_c if best_d is not None and best_d <= spacing * 0.35 else None
            return out

        map_v = index_map(clusters_v, fitted_v, spacing_v)
        map_h = index_map(clusters_h, fitted_h, spacing_h)

        board_pts, image_pts = [], []
        for i in range(n):  # row / horizontal-line index
            if map_h[i] is None:
                continue
            rho_h, theta_h = map_h[i]
            for j in range(n):  # col / vertical-line index
                if map_v[j] is None:
                    continue
                rho_v, theta_v = map_v[j]
                pt = _line_intersection(rho_v, theta_v, rho_h, theta_h)
                if pt is None:
                    continue
                board_pts.append([j, i])
                image_pts.append(pt)

        if len(board_pts) < 4:
            last_err = f"too few observed line intersections ({len(board_pts)}) at threshold {thresh}"
            continue

        board_pts = np.array(board_pts, dtype=np.float32)
        image_pts = np.array(image_pts, dtype=np.float32)
        H, mask = cv2.findHomography(board_pts, image_pts, cv2.RANSAC, ransacReprojThreshold=4.0)
        if H is None:
            last_err = f"homography fit failed at threshold {thresh}"
            continue

        mask = mask.ravel().astype(bool)
        n_inliers = int(mask.sum())
        reproj = cv2.perspectiveTransform(board_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj_err = float(np.linalg.norm(reproj[mask] - image_pts[mask], axis=1).mean()) if n_inliers else 1e9

        spacing_ratio = min(spacing_v, spacing_h) / max(spacing_v, spacing_h)  # 1.0 = perfectly consistent
        score = (n_inliers / (1.0 + reproj_err)) * spacing_ratio

        # Map the ideal uniform 9x9 board grid through H to get final image points.
        ideal = np.array([[j, i] for i in range(n) for j in range(n)], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(ideal, H).reshape(n, n, 2)
        grid = mapped.astype(np.float32)

        # Penalize extrapolation that falls well outside the source image --
        # a chessboard is normally fully framed in the shot, so a homography
        # whose corners land noticeably off-canvas is a sign it was fit from
        # too narrow/lopsided a set of observed lines (e.g. one outer edge
        # was cropped away), even if its reprojection error on the points it
        # did see looks good. This is a soft, smoothly-decaying penalty
        # rather than a hard cutoff so mild, legitimate edge-of-frame boards
        # aren't unfairly rejected.
        img_h, img_w = gray.shape
        max_dim = max(img_w, img_h)
        overshoot = max(0.0,
                         -float(grid[:, :, 0].min()), float(grid[:, :, 0].max()) - img_w,
                         -float(grid[:, :, 1].min()), float(grid[:, :, 1].max()) - img_h) / max_dim
        bounds_penalty = max(0.0, 1.0 - overshoot / 0.08)  # full penalty by 8% overshoot
        score *= bounds_penalty

        candidates.append((score, grid, dict(thresh=thresh, n_inliers=n_inliers,
                                              reproj_err=reproj_err, spacing_ratio=spacing_ratio,
                                              overshoot=overshoot)))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates


def _find_grid_candidates_scaled(img, n=9, hough_thresholds=(140, 120, 100, 80, 60), target_max_dim=750):
    """
    Wraps _find_grid_candidates with resolution normalization: a modern
    phone photo can be 3000+px on a side, and the Hough vote threshold /
    clustering tolerances in _find_grid_candidates were calibrated in pixel
    units, so run at very high resolution they silently stop meaning what
    they meant when tuned (observed on a 1280x1275 real-photo test case:
    over 40,000 Hough lines even at the strictest threshold, versus a few
    hundred on similarly-composed smaller images -- the clustering and
    9-line progression search effectively drown in noise). Downscaling to a
    common working resolution before detection, then scaling the resulting
    grid coordinates back up, makes the whole detector resolution-invariant
    without having to re-tune any of its internal constants per image size.
    """
    h, w = img.shape[:2]
    scale = min(1.0, target_max_dim / max(h, w))
    small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    candidates = _find_grid_candidates(small, n=n, hough_thresholds=hough_thresholds)
    if scale < 1.0:
        candidates = [(score, grid / scale, info) for score, grid, info in candidates]
    return candidates


def _order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _find_quad_color(img):
    """
    Tier-2 fallback detector: find the board via color/saturation instead
    of lines. Don't threshold the whole board into one solid mask (that
    forces a big morphological closing, which easily bridges a thin gap to
    UI text/chrome sitting right above the board) -- threshold only the
    *saturated* squares (the colored ones in most digital board themes),
    which naturally form isolated blobs tiled across the board with no
    closing needed, then take the convex hull of all of those blobs. Since
    the blobs are periodic and span the full board, their hull's corners
    land almost exactly on the board's outer corners, independent of
    whatever surrounds the board and independent of line/edge quality --
    this is what makes it a genuine fallback for images where piece artwork
    and on-board text overwhelm the line-based detector (see
    PIPELINE_LOG.md's tough.jpg case).

    Returns 4 ordered corners [TL,TR,BR,BL], or None.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > 30) & (v > 60)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = img.shape[0] * img.shape[1]
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * img_area:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps_frac in [0.01, 0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2))
    rect = cv2.minAreaRect(hull)
    return _order_corners(cv2.boxPoints(rect))


def _find_quad_color_rect(img):
    """
    Variant of _find_quad_color for axis-aligned (or near axis-aligned)
    boards. approxPolyDP can under-shoot a corner when the color mask has a
    small gap right at that corner square, landing a full square short of
    the true corner. cv2.minAreaRect instead fits the minimal rectangle
    enclosing every hull point, so as long as the true top and left edges
    are each individually represented *somewhere* in the hull (even never
    at the same vertex), the fitted rectangle's corner still lands right.
    Only valid for a non-trapezoidal board, so it's a separate candidate
    rather than a replacement.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > 30) & (v > 60)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = img.shape[0] * img.shape[1]
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * img_area:
        return None
    rect = cv2.minAreaRect(c)
    return _order_corners(cv2.boxPoints(rect))


def _find_quad_edges(img):
    """Tier-3 fallback: largest convex 4-point contour in the Canny edge map."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    best, best_area = None, -1
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area >= 0.05 * img_area and area > best_area:
                best_area, best = area, approx.reshape(4, 2)
    if best is None:
        return None
    return _order_corners(best)


def _quad_sanity_score(quad, img_shape):
    """
    Reward large, roughly-square quads (a chessboard is ~1:1 aspect even
    under perspective), and flag ones that hug the outer image frame on 2+
    sides -- a common failure where the detector grabs a decorative
    border/UI frame instead of just the 64-square grid inside it. Returns
    (score, is_border_hug); score <= 0 means reject.
    """
    h, w = img_shape[:2]
    area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))
    if area < 0.1 * h * w:
        return -1.0, True

    tl, tr, br, bl = quad
    avg_horiz = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    avg_vert = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if avg_vert == 0:
        return -1.0, True
    aspect_penalty = abs(avg_horiz / avg_vert - 1.0)
    score = area * max(0.0, 1.0 - aspect_penalty)

    tol = 0.01 * min(h, w)
    xs, ys = quad[:, 0], quad[:, 1]
    touches = sum([xs.min() < tol, xs.max() > w - tol, ys.min() < tol, ys.max() > h - tol])
    is_border_hug = touches >= 2
    if is_border_hug:
        score *= 0.15
    return score, is_border_hug


def _grid_alignment_score(img, grid, size=400):
    """
    Post-hoc validity check, used only as a same-strategy tie-breaker (see
    caveat below) -- NOT a cross-strategy arbiter. Warps the candidate grid
    to a square canvas and measures what fraction of total gradient-energy
    falls right on the 7 interior grid lines vs. elsewhere: a correctly
    aligned grid concentrates real color-transition edges there, while a
    misaligned one smears that energy off-grid.

    CAVEAT (found empirically, see PIPELINE_LOG.md): on a low-contrast
    board with detailed piece artwork, this score can be *fooled* -- a
    shifted/misaligned quad scored higher than the true fit on one of our
    test images, because piece-outline edges are much stronger than the
    faint true square-to-square edges and can coincidentally land on the
    measured bands under the wrong alignment. That's why this is only used
    to break ties within a single detector strategy's own candidates
    (which are already validated by that strategy's own internal checks),
    never to pick a winner across fundamentally different strategies.
    """
    tl, tr, br, bl = grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    try:
        M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    except cv2.error:
        return -1.0
    warped = cv2.warpPerspective(img, M, (size, size))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    sq = size / 8.0
    band = max(2, int(size * 0.01))
    on_grid = 0.0
    for i in range(1, 8):
        c = int(round(i * sq))
        lo, hi = max(0, c - band), min(size, c + band)
        on_grid += mag[lo:hi, :].sum()
        on_grid += mag[:, lo:hi].sum()
    return on_grid / (mag.sum() + 1e-6)


def _trim_uniform_frame(img, corners, n=9, max_trim_steps=6, threshold_frac=0.04, min_threshold=20):
    """
    Tier 2 (and 3)'s quad detectors identify the board by color/edge signal
    alone, with no way to distinguish a genuine checkered square from a
    decorative frame of similar color/texture surrounding coordinate
    labels. Confirmed on a real test image: ~15-30px of uniform wood-toned
    frame sat between the coordinate labels and the true checkered
    squares, and Tier 2 included it wholesale in the detected quad, since
    the frame satisfies the same saturation threshold with nothing to
    distinguish it from a real board square.

    Fix: check each of the 4 edges' outermost ring of (n-1) cells for real
    cross-cell color variance -- a genuine checkered row/column alternates
    in brightness; a uniform frame does not. If an edge's outer ring is
    suspiciously uniform, shift that edge inward (linearly interpolated
    along the edge, in small quarter-cell steps to avoid overshooting a
    thin frame) and re-check, stopping the moment real alternation is
    found.

    threshold_frac is ADAPTIVE, not a fixed absolute cutoff: a fixed
    number doesn't generalize across board themes with very different
    native contrast. Confirmed directly: a high-contrast wood-tone board
    showed ~10-17 variance on its frame vs 300-3000+ once past it into
    real squares (a fixed threshold of 150 worked fine there), but a
    low-contrast pastel-blue board showed genuine real alternation at only
    26-92 variance -- the SAME fixed threshold of 150 incorrectly kept
    trimming straight through that real content, stopping ~15px too late
    (into board, not frame) because it never saw a ring above 150 until it
    coincidentally hit a piece-silhouette edge. Fix: measure this specific
    board's own safely-interior cross-cell variance once (a middle row and
    middle column, far from any edge effects) and use a small fraction of
    THAT as the per-axis threshold instead. Confirmed on both real test
    cases: frame variance is consistently <2% of interior variance while
    genuine (even low-contrast) board content is consistently >6%, so
    threshold_frac=0.04 sits with real margin on both sides -- still only
    validated against two real images, not a large calibrated set.
    """
    corners = corners.astype(np.float32).copy()

    def sample_ring_variance(quad, edge, offset_px=5, patch_half=3):
        """
        Sample a small 2D patch (patch_half*2+1 square, to average out
        local wood-grain/texture noise) at a fixed inward offset from the
        tested edge, at n-1 evenly-spaced points along that edge, directly
        from the source image.

        Two prior versions of this function each had a different bug on
        real test images, both now fixed by this design:
          1. A patch sized as a FRACTION of the current (possibly still
             oversized, frame+content-straddling) cell -- on the very
             first, largest cell, that patch could already reach past the
             frame into real content and trigger a false-early stop
             several px before the true edge.
          2. A single-pixel-wide 1D sampling line at a fixed offset --
             fixed the above, but with no spatial averaging at all, local
             wood-grain texture noise within the STILL-uniform frame was
             enough to read as "real content" and stop trimming before any
             progress was made.
        A small averaged 2D patch at a fixed, small inward offset avoids
        both: precisely positioned (unlike #1), but smoothed enough to not
        be fooled by single-pixel texture noise (unlike #2).
        """
        tl, tr, br, bl = quad
        if edge == 'top':
            p0, p1, inward = tl, tr, bl - tl
        elif edge == 'bottom':
            p0, p1, inward = bl, br, tl - bl
        elif edge == 'left':
            p0, p1, inward = tl, bl, tr - tl
        else:
            p0, p1, inward = tr, br, tl - tr
        inward = inward / (np.linalg.norm(inward) + 1e-6)

        means = []
        for i in range(n - 1):
            base = p0 + (p1 - p0) * (i + 0.5) / (n - 1) + inward * offset_px
            cx, cy = int(round(base[0])), int(round(base[1]))
            y0, y1 = max(0, cy - patch_half), min(img.shape[0], cy + patch_half + 1)
            x0, x1 = max(0, cx - patch_half), min(img.shape[1], cx + patch_half + 1)
            if y1 > y0 and x1 > x0:
                means.append(float(img[y0:y1, x0:x1].astype(np.float32).mean()))
        return float(np.var(means)) if len(means) >= 2 else 0.0

    # Adaptive reference: safely-interior middle row/column, far from any
    # edge/frame effects, sampled the same fixed-thin-band way as the ring
    # checks (not via a resized per-cell patch) so the comparison is
    # genuinely apples-to-apples.
    def sample_mid_variance(quad, axis, patch_half=3):
        tl, tr, br, bl = quad
        if axis == 'row':  # a horizontal line at the middle row, varying by column
            p0 = tl + (bl - tl) * 0.5
            p1 = tr + (br - tr) * 0.5
        else:  # 'col': a vertical line at the middle column, varying by row
            p0 = tl + (tr - tl) * 0.5
            p1 = bl + (br - bl) * 0.5
        means = []
        for i in range(n - 1):
            base = p0 + (p1 - p0) * (i + 0.5) / (n - 1)
            cx, cy = int(round(base[0])), int(round(base[1]))
            y0, y1 = max(0, cy - patch_half), min(img.shape[0], cy + patch_half + 1)
            x0, x1 = max(0, cx - patch_half), min(img.shape[1], cx + patch_half + 1)
            if y1 > y0 and x1 > x0:
                means.append(float(img[y0:y1, x0:x1].astype(np.float32).mean()))
        return float(np.var(means)) if len(means) >= 2 else 0.0

    row_ref = sample_mid_variance(corners, 'row')
    col_ref = sample_mid_variance(corners, 'col')
    row_threshold = max(min_threshold, row_ref * threshold_frac)  # for 'top'/'bottom' (vary by column)
    col_threshold = max(min_threshold, col_ref * threshold_frac)  # for 'left'/'right' (vary by row)

    # Quarter-cell steps, not full-cell: an observed frame (~30px) was well
    # under half a cell width (~73px) on the test image that surfaced this.
    # A full-cell step would overshoot straight through the frame and eat
    # into 1-2 real ranks/files.
    step_frac = 1.0 / (n * 4)
    for edge in ['top', 'bottom', 'left', 'right']:
        threshold = row_threshold if edge in ('top', 'bottom') else col_threshold
        for _ in range(max_trim_steps):
            if sample_ring_variance(corners, edge) >= threshold:
                break
            tl, tr, br, bl = corners
            if edge == 'top':
                corners = np.array([tl + (bl - tl) * step_frac, tr + (br - tr) * step_frac, br, bl], dtype=np.float32)
            elif edge == 'bottom':
                corners = np.array([tl, tr, br + (tr - br) * step_frac, bl + (tl - bl) * step_frac], dtype=np.float32)
            elif edge == 'left':
                corners = np.array([tl + (tr - tl) * step_frac, tr, br, bl + (br - bl) * step_frac], dtype=np.float32)
            else:
                corners = np.array([tl, tr + (tl - tr) * step_frac, br + (bl - br) * step_frac, bl], dtype=np.float32)

    return corners


def _refine_quad_via_edge_peaks(img, quad, n=9, search_radius=15):
    """
    Precision refinement pass, run after _trim_uniform_frame. That coarse
    trim's stopping rule (a single local cross-cell variance check against
    a fixed threshold) is fragile: confirmed on a real test image, it
    correctly did ~90% of the needed correction on the top edge (28.8px
    true error -> 3.2px residual) but left a much larger residual on the
    other three edges (~8-10px) -- most likely because a rank/file label
    digit or other incidental texture sitting near the true boundary
    triggered a false "real content, stop trimming" reading on a single
    local check before reaching the actual checkered square edge.

    Fix: instead of one local pass/fail check per edge, find the strongest
    Canny edge-density peak near EACH of the current 9 assumed grid line
    positions independently (a small +-search_radius window per line),
    then fit a single linear regression (uniform spacing model) through
    all 9 peaks per axis. Averaging over all 9 lines via least-squares is
    naturally robust to any one line's peak being thrown off by nearby
    text/piece artwork, unlike a single local check -- confirmed this
    recovers the true boundary to within ~1px of an independently
    hand-verified reference on the same test image.

    Only valid for axis-aligned or near-axis-aligned quads (this tier's
    existing target case) -- assumes the true grid lines are straight and
    parallel to the image axes near the current fit, which is reasonable
    once _trim_uniform_frame has already gotten close. Falls back to
    returning the input quad unchanged (never raises) if the geometry is
    too degenerate to search.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    tl, tr, br, bl = quad
    x_lo = int(max(0, min(tl[0], bl[0]) + 10))
    x_hi = int(min(w, max(tr[0], br[0]) - 10))
    y_lo = int(max(0, min(tl[1], tr[1]) + 10))
    y_hi = int(min(h, max(bl[1], br[1]) - 10))
    if x_hi <= x_lo or y_hi <= y_lo:
        return quad

    row_density = edges[:, x_lo:x_hi].sum(axis=1).astype(float)
    col_density = edges[y_lo:y_hi, :].sum(axis=0).astype(float)

    def refine_axis(density, positions):
        """
        Returns (intercept, slope) fit through only the reference positions
        that have a genuine, above-noise-floor edge peak nearby.

        Critical case this guards against: on a board whose internal grid
        lines are too low-contrast for Canny to detect at all (confirmed
        directly -- most interior lines showed density=0 across their
        entire +-search_radius window on a real low-contrast test board),
        np.argmax on an all-zero window silently returns index 0 of that
        window, i.e. exactly `ref - search_radius` every time. That's not
        a real peak, just an artifact -- but blindly regressing through
        several such fake points (which all share the same systematic
        `-search_radius` offset pattern) badly corrupts the fit, dragging
        even the genuinely strong endpoints (e.g. the true outer boundary,
        which DOES have real signal) to the wrong answer. Only the board's
        two outer edges reliably have strong signal on such a board (a
        real border-to-background transition), so this must tolerate
        fitting through as few as 2 valid points.
        """
        peaks, valid_idx, peak_densities = [], [], []
        for j, ref in enumerate(positions):
            lo = max(0, int(ref - search_radius))
            hi = min(len(density), int(ref + search_radius))
            if hi <= lo:
                continue
            window = density[lo:hi]
            if window.max() <= 0:
                continue  # no real edge signal in this window at all -- skip, don't fake a peak
            peaks.append(lo + int(np.argmax(window)))
            valid_idx.append(j)
            peak_densities.append(window.max())

        if len(peaks) < 2:
            return None

        # Among valid peaks, also drop any whose density is negligible
        # relative to the strongest one found (a real edge should be
        # within an order of magnitude of the strongest; anything far
        # weaker is more likely texture noise than a true grid line).
        max_density = max(peak_densities)
        keep = [(idx, p) for idx, p, d in zip(valid_idx, peaks, peak_densities)
                if d >= max_density * 0.05]
        if len(keep) < 2:
            return None
        valid_idx, peaks = zip(*keep)

        idx = np.array(valid_idx, dtype=float)
        A = np.vstack([idx, np.ones(len(idx))]).T
        slope, intercept = np.linalg.lstsq(A, np.array(peaks, dtype=float), rcond=None)[0]
        return intercept, slope

    grid = grid_from_corners(quad, n=n)
    row_fit = refine_axis(row_density, grid[:, n // 2, 1])
    col_fit = refine_axis(col_density, grid[n // 2, :, 0])
    if row_fit is None or col_fit is None:
        return quad

    row0, row_step = row_fit
    col0, col_step = col_fit
    row_last = row0 + (n - 1) * row_step
    col_last = col0 + (n - 1) * col_step

    return np.array([[col0, row0], [col_last, row0],
                      [col_last, row_last], [col0, row_last]], dtype=np.float32)


def _edge_undershoot_penalty(img, quad, n=9, outward_px=12, patch_half=10):
    """
    Post-hoc validity check for candidate SELECTION (hull vs rect, etc.),
    distinct from and complementary to _trim_uniform_frame's inward checks
    and _grid_alignment_score's on-grid-energy check.

    Confirmed necessary on a real test image where BOTH existing selection
    signals (_quad_sanity_score AND _grid_alignment_score) picked the
    WRONG candidate: one quad-detector variant's trim+refine correctly
    landed on the true outer board edge, while another variant's landed
    ~18px short, on a strong INTERNAL grid line (the boundary between two
    ranks) that happened to look like a plausible edge to both of those
    metrics -- the internal line is real, strong, and can coincidentally
    align well in a warped-canvas gradient-energy measurement, even though
    it's the wrong line entirely.

    This check asks a more direct question: is there still genuine
    board-like content sitting just OUTSIDE the candidate boundary? A
    correct boundary should have uniform frame/background out there; an
    undershot one (stopped at an internal line instead of the true edge)
    will still show real cross-cell color variance just past it. Returns
    a fraction in [0, 1] -- 0 means no undershoot detected, 1 means all 4
    edges show strong content just outside (fully rejected candidate).
    """
    tl, tr, br, bl = quad

    def sample_variance_at(p0, p1, direction, offset):
        means = []
        for i in range(n - 1):
            base = p0 + (p1 - p0) * (i + 0.5) / (n - 1) + direction * offset
            cx, cy = int(round(base[0])), int(round(base[1]))
            y0, y1 = max(0, cy - patch_half), min(img.shape[0], cy + patch_half + 1)
            x0, x1 = max(0, cx - patch_half), min(img.shape[1], cx + patch_half + 1)
            if y1 > y0 and x1 > x0:
                means.append(float(img[y0:y1, x0:x1].astype(np.float32).mean()))
        return float(np.var(means)) if len(means) >= 2 else 0.0

    # "outward" = away from board center, i.e. the negative of each edge's
    # own inward direction used in _trim_uniform_frame.
    checks = [
        (tl, tr, (tl - bl) / (np.linalg.norm(tl - bl) + 1e-6)),  # top, outward = up
        (bl, br, (bl - tl) / (np.linalg.norm(bl - tl) + 1e-6)),  # bottom, outward = down
        (tl, bl, (tl - tr) / (np.linalg.norm(tl - tr) + 1e-6)),  # left, outward = left
        (tr, br, (tr - tl) / (np.linalg.norm(tr - tl) + 1e-6)),  # right, outward = right
    ]

    # Reference: genuine interior content variance, so "real content found
    # outside" is judged relative to this board's own contrast level, not
    # an absolute number (same adaptive-threshold reasoning as
    # _trim_uniform_frame -- see that function's docstring for why a fixed
    # absolute threshold doesn't generalize across board themes).
    mid_top = tl + (tr - tl) * 0.5
    mid_bot = bl + (br - bl) * 0.5
    interior_ref = sample_variance_at(mid_top, mid_bot, np.array([0.0, 0.0]), 0)
    threshold = max(20.0, interior_ref * 0.04)

    flags = []
    for p0, p1, outward in checks:
        v = sample_variance_at(p0, p1, outward, outward_px)
        flags.append(1.0 if v >= threshold else 0.0)
    return sum(flags) / len(flags)


def _find_grid_via_quad(img, quad_fn, n=9):
    """
    Run a quad-detector function, validate it (_quad_sanity_score, plus the
    same bounds-overshoot logic as the line-based detector), and if valid,
    return (score, grid, info) -- else None.
    """
    try:
        quad = quad_fn(img)
    except Exception:
        quad = None
    if quad is None:
        return None
    sanity, is_border_hug = _quad_sanity_score(quad, img.shape)
    if sanity <= 0:
        return None

    quad = _trim_uniform_frame(img, quad, n=n)
    quad = _refine_quad_via_edge_peaks(img, quad, n=n)
    grid = grid_from_corners(quad, n=n)

    img_h, img_w = img.shape[:2]
    max_dim = max(img_w, img_h)
    overshoot = max(0.0,
                     -float(grid[:, :, 0].min()), float(grid[:, :, 0].max()) - img_w,
                     -float(grid[:, :, 1].min()), float(grid[:, :, 1].max()) - img_h) / max_dim
    bounds_penalty = max(0.0, 1.0 - overshoot / 0.08)
    if bounds_penalty <= 0:
        return None

    align = _grid_alignment_score(img, grid)
    undershoot = _edge_undershoot_penalty(img, quad, n=n)
    # Undershoot is checked AFTER trim+refine on the FINAL quad -- a real,
    # direct signal that a candidate's boundary stopped short of the true
    # edge (see _edge_undershoot_penalty docstring). Applied as a hard
    # multiplicative penalty, not folded into the tie-break alone: on the
    # real case that surfaced this, both sanity and alignment already
    # favored the undershot candidate, so a tie-break addition wouldn't
    # have been enough to flip the outcome.
    score = sanity * bounds_penalty * (0.15 if is_border_hug else 1.0) * (1.0 - 0.85 * undershoot)
    return score, grid, dict(sanity=sanity, is_border_hug=is_border_hug,
                              overshoot=overshoot, alignment=align, undershoot=undershoot)


def _contrast_stretch(gray):
    """Global min-max contrast stretch. Deliberately not CLAHE here: CLAHE's
    local tiling re-introduces noise on genuinely flat regions, whereas a
    single linear stretch across the whole dynamic range is enough when the
    entire board shares one narrow band of grey values (adjacent squares
    only ~14 grey levels apart in the hardest case seen so far)."""
    mn, mx = gray.min(), gray.max()
    if mx <= mn:
        return gray.copy()
    return ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)


def _refine_bbox_edge(profile, coarse_idx, direction, search=15):
    """
    Stylized board cards often carry a soft drop-shadow that bleeds several
    px past the true edge, worst right at the bottom-right corner. A plain
    "first index back at background level" threshold walks into that
    gradual fade and overshoots. Search a small window around the coarse
    threshold crossing and pick the point of steepest intensity change: the
    shadow fades gradually (small, smooth gradient per pixel) while the
    board's real boundary is a comparatively sharp step. direction=+1 for
    an edge whose background lies at increasing index (right/bottom), -1
    for decreasing index (left/top).
    """
    lo = max(0, coarse_idx - search)
    hi = min(len(profile) - 1, coarse_idx + search)
    seg = profile[lo:hi + 1]
    if len(seg) < 2:
        return coarse_idx
    grad = np.diff(seg) * direction
    k = int(np.argmax(grad))
    return lo + k + 1


def _find_board_bbox_axis_aligned(gray_stretched):
    """
    Tier-4 fallback: find the board's axis-aligned bounding box from ROW/
    COLUMN MEAN brightness profiles, rather than from any internal grid
    structure. This is the key difference from Tiers 1-3, and what makes it
    a genuine fallback for the lowest-signal case seen so far (a board
    that's mostly empty squares at ~14-grey-level contrast, with almost no
    saturation and too few pieces to give Tier 1 enough line evidence, or
    Tier 2 enough color evidence, along several of the 9 grid lines):
    averaging brightness across an entire row/column cancels per-pixel
    noise and reveals a clear step between "background outside the board"
    and "board interior" even when individual pixels barely differ,
    *without needing any signal from the interior 8x8 structure at all* --
    only the outer silhouette against a plain background.

    Necessarily assumes the board is axis-aligned in the image (a flat
    screenshot, not a perspective photo) and exactly square -- there is no
    perspective correction here, unlike Tiers 1-3. That's an acceptable
    trade for a last-resort tier: it only ever runs after Tiers 1-3 have
    already failed to find anything.

    Returns (x0, y0, x1, y1) or raises if no plausible edge is found (the
    caller is expected to catch this -- this function has no cross-image
    robustness guarantee, it's tuned for "plain background around a
    low-contrast card").
    """
    h, w = gray_stretched.shape

    col_mean = gray_stretched.mean(axis=0)
    plateau = np.median(col_mean[w // 4: 3 * w // 4])
    half = (255 + plateau) / 2

    cols_below = np.where(col_mean < half)[0]
    x0_coarse, x1_coarse = int(cols_below.min()), int(cols_below.max()) + 1
    x0 = _refine_bbox_edge(col_mean, x0_coarse, direction=-1)
    x1 = _refine_bbox_edge(col_mean, x1_coarse, direction=+1)
    size = x1 - x0

    # top edge from a strip on the side away from any corner decoration/chrome
    strip = gray_stretched[:, w - 100: w - 20]
    row_mean = strip[: h // 3].mean(axis=1)  # only look near the top
    rows_below = np.where(row_mean < half)[0]
    y0 = int(rows_below.min())
    y1 = y0 + size

    return x0, y0, x1, y1


def _find_grid_via_bbox(img, n=9):
    """
    Run the Tier-4 axis-aligned bbox detector and, if it produces a
    plausible result, convert it to the standard (9,9,2) grid via
    grid_from_corners. Returns (score, grid, info) or None.

    Deliberately conservative validity check: reject anything not close to
    square (a chessboard is 1:1) or implausibly small, since this method
    has no internal cross-check (no RANSAC inliers, no line-count) the way
    Tiers 1-3 do -- its only self-consistency signal is "is the box roughly
    the right shape".
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stretched = _contrast_stretch(gray)
        x0, y0, x1, y1 = _find_board_bbox_axis_aligned(stretched)
    except Exception:
        return None

    h, w = img.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    if bw < 0.15 * w or bh < 0.15 * h:
        return None
    aspect_penalty = abs(bw / bh - 1.0)
    if aspect_penalty > 0.15:
        return None
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    grid = grid_from_corners(quad, n=n)
    align = _grid_alignment_score(img, grid)
    score = (bw * bh) * max(0.0, 1.0 - aspect_penalty)
    return score, grid, dict(bbox=(x0, y0, x1, y1), aspect_penalty=aspect_penalty, alignment=align)


def _periodic_grid_lines_1d(profile, dim, n_lines=9, spacing_search_frac=0.35, refine_radius=3):
    """
    Find n_lines periodic grid-line positions along one axis by searching
    for the strongest evenly-spaced ("comb") pattern in an edge-density
    profile, via FFT autocorrelation for a coarse spacing estimate over a
    small neighborhood (not just the single coarse peak -- a spacing that's
    off by 1-2px can leave too little phase slack to reach the true origin
    if it sits close to the search boundary), then subpixel end-anchoring
    (snap the first/last line to the true nearest edge-density peak, then
    linearly interpolate the interior lines between those two anchors,
    since the true spacing is usually fractional and an integer-pixel comb
    search alone can't satisfy strong edges at both ends at once).

    Returns (positions[n_lines], spacing, raw_comb_score) or None.
    Ported from periodic_grid.py (see that file for the fuller derivation/
    rationale and the real test cases that shaped each step).
    """
    def comb_score(profile, spacing, phase, n_lines):
        idx = np.round(phase + spacing * np.arange(n_lines)).astype(int)
        valid = (idx >= 0) & (idx < len(profile))
        if valid.sum() < n_lines:
            return -1
        return profile[idx].sum()

    def best_spacing_via_autocorr(profile, lo, hi):
        p = profile - profile.mean()
        n = len(p)
        f = np.fft.rfft(p, n=2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n]
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return None
        return lo + int(np.argmax(acf[lo:hi]))

    def snap_to_peak(pos, profile, radius):
        lo = max(0, int(pos) - radius)
        hi = min(len(profile), int(pos) + radius + 1)
        if hi <= lo or profile[lo:hi].max() <= 0:
            return pos
        return float(lo + np.argmax(profile[lo:hi]))

    lo = dim / 8 * (1 - spacing_search_frac)
    hi = dim / 8 * (1 + spacing_search_frac)
    spacing0 = best_spacing_via_autocorr(profile, lo, hi)
    if spacing0 is None or spacing0 <= 0:
        return None

    best_overall = None
    for spacing in range(max(1, spacing0 - refine_radius), spacing0 + refine_radius + 1):
        max_phase = dim - spacing * (n_lines - 1)
        if max_phase < 0:
            continue
        best_phase, best_score = 0, -1
        for phase in range(0, int(max_phase) + 1):
            s = comb_score(profile, spacing, phase, n_lines)
            if s > best_score:
                best_score, best_phase = s, phase
        if best_overall is None or best_score > best_overall[2]:
            best_overall = (spacing, best_phase, best_score)

    if best_overall is None:
        return None
    spacing, best_phase, best_score = best_overall
    positions = best_phase + spacing * np.arange(n_lines)

    snap_radius = max(8, int(spacing * 0.3))
    first_anchor = snap_to_peak(positions[0], profile, snap_radius)
    last_anchor  = snap_to_peak(positions[-1], profile, snap_radius)
    if last_anchor > first_anchor:
        positions = first_anchor + (last_anchor - first_anchor) * np.arange(n_lines) / (n_lines - 1)

    return positions.astype(float), spacing, best_score


def _find_grid_via_periodicity(img, n=9):
    """
    Tier 5: axis-aligned periodicity detector. Ported from a separate
    prototype (periodic_grid.py) built and validated independently on 6
    clean digital-board renders before this tiered pipeline existed.

    Distinct signal from every other tier: instead of requiring individual
    grid *lines* to out-vote noise in a Hough transform (Tier 1), a
    saturated color mask (Tier 2), a clean outer contour (Tier 3), or a
    plain-background brightness step (Tier 4), this looks for the
    checkerboard's own defining global signature -- edge density that
    repeats roughly every image_dimension/8 pixels, found via FFT
    autocorrelation -- which can survive in some images where none of the
    other four signals individually do. Like Tier 4, this assumes an
    axis-aligned board (no perspective correction), so it's included as a
    last-resort addition, not a replacement for any earlier tier.

    NOTE: a simpler version of this idea (single autocorrelation peak, no
    comb-matching or subpixel end-anchoring) was already tried directly in
    this pipeline and failed on the sparse/mostly-empty test board (see
    PIPELINE_LOG.md SS10) -- there was no periodic signal at all in the
    fully-empty rows for autocorrelation to find. This fuller version is
    NOT expected to fix that specific case (same fundamental limitation:
    no signal source, not a weak search), but is included as further
    defense-in-depth for OTHER images that might reach Tier 5 in the
    future with a different failure profile than what's been seen so far.

    Scored using this method's OWN internal evidence (how far the matched
    comb positions' edge density stands above that axis's noise floor,
    i.e. a z-score-like measure) -- never compared against another tier's
    score (see find_board_grid_robust's docstring for why that's unsafe).

    Returns (score, grid, info) or None.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        col_density = edges.sum(axis=0).astype(float)
        row_density = edges.sum(axis=1).astype(float)

        x_result = _periodic_grid_lines_1d(col_density, w, n)
        y_result = _periodic_grid_lines_1d(row_density, h, n)
        if x_result is None or y_result is None:
            return None
        x_lines, x_spacing, x_comb = x_result
        y_lines, y_spacing, y_comb = y_result
    except Exception:
        return None

    if x_lines[0] < -5 or y_lines[0] < -5 or x_lines[-1] > w + 5 or y_lines[-1] > h + 5:
        return None
    if max(x_spacing, y_spacing) <= 0:
        return None
    spacing_ratio = min(x_spacing, y_spacing) / max(x_spacing, y_spacing)
    if spacing_ratio < 0.7:  # too far from square to plausibly be a chessboard
        return None

    quad = np.array([[x_lines[0], y_lines[0]], [x_lines[-1], y_lines[0]],
                      [x_lines[-1], y_lines[-1]], [x_lines[0], y_lines[-1]]], dtype=np.float32)
    grid = grid_from_corners(quad, n=n)
    align = _grid_alignment_score(img, grid)

    x_z = (x_comb / n - col_density.mean()) / (col_density.std() + 1e-6)
    y_z = (y_comb / n - row_density.mean()) / (row_density.std() + 1e-6)
    score = min(x_z, y_z) * spacing_ratio
    # A bare `score > 0` was inconsistent on pure noise across repeated runs
    # (sometimes marginally positive, sometimes not) -- require the weaker
    # axis to clear a real margin above the noise floor, not just barely
    # exceed it. Untested against a large adversarial set; a placeholder
    # improvement, not a rigorously calibrated cutoff (consistent with this
    # method's other constants ported from the periodic_grid.py prototype).
    if score <= 1.5:
        return None

    return score, grid, dict(x_spacing=x_spacing, y_spacing=y_spacing,
                              spacing_ratio=spacing_ratio, alignment=align)


def _measure_true_tilt(img, near_angle_tol_deg=20):
    """
    Independently measure the image's true dominant near-horizontal and
    near-vertical line tilt via a robust median across many detected Hough
    lines -- deliberately NOT reusing Tier 1's own line-clustering/
    homography-fit path, so this serves as an independent check rather
    than re-deriving the same (possibly slightly biased) answer.

    Confirmed necessary on a real photographed test image: Tier 1's fitted
    grid edges measured ~0.5deg tilt, while this independent Hough-median
    measurement found ~1.0deg -- a small but real and visually-confirmed
    undershoot (the user could see the grid needed "very very little more
    tilt to the left"), producing a few pixels of cumulative corner-drift
    on a ~600px board.

    Returns (h_tilt_deg, v_tilt_deg) -- median offsets from perfectly
    horizontal (0deg) and perfectly vertical (90deg) respectively -- or
    (None, None) if too few lines were found to measure reliably.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=100)
    if lines is None or len(lines) < 6:
        lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=70)
    if lines is None or len(lines) < 6:
        return None, None

    h_offsets, v_offsets = [], []
    for l in lines:
        rho, theta = l[0]
        deg = np.degrees(theta)
        if deg < near_angle_tol_deg or deg > 180 - near_angle_tol_deg:
            v_offsets.append(deg if deg < 90 else deg - 180)
        elif abs(deg - 90) < near_angle_tol_deg:
            h_offsets.append(deg - 90)

    if len(h_offsets) < 3 or len(v_offsets) < 3:
        return None, None
    return float(np.median(h_offsets)), float(np.median(v_offsets))


def _refine_grid_tilt(img, corners, n=9, max_correction_deg=3.0):
    """
    Post-hoc rotation refinement for Tier 1's output: the RANSAC
    homography/line-clustering fit can slightly under- or over-estimate
    the board's true overall tilt even when it gets translation/spacing
    right (confirmed on a real photographed test image -- see
    _measure_true_tilt's docstring).

    Rigidly rotates the current quad about its own center to match the
    independently-measured true tilt -- a small correction to an
    already-close fit, not a re-detection. Bounded by max_correction_deg
    and only applied when both the current quad's own implied tilt and
    the independently-measured true tilt roughly agree on direction/
    magnitude already (within a factor of 3) -- if they disagree wildly,
    that signals the independent measurement itself is unreliable for
    this image (e.g. a busy/noisy background), and blindly trusting it
    would risk making a good fit worse rather than better.
    """
    tl, tr, br, bl = corners
    current_h_tilt = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    current_v_tilt = np.degrees(np.arctan2(bl[1] - tl[1], bl[0] - tl[0])) - 90

    true_h_tilt, true_v_tilt = _measure_true_tilt(img)
    if true_h_tilt is None:
        return corners

    delta_h = true_h_tilt - current_h_tilt
    delta_v = true_v_tilt - current_v_tilt
    delta = float(np.median([delta_h, delta_v]))

    if abs(delta) > max_correction_deg:
        return corners  # too large a correction to trust blindly -- leave as-is
    # Sanity: current and true tilt should roughly agree on sign/rough
    # magnitude already (this is a refinement, not a rescue) -- if the
    # independent measurement implies a wildly different direction, don't
    # apply it.
    if abs(current_h_tilt) > 0.5 and np.sign(current_h_tilt) != np.sign(true_h_tilt):
        return corners

    center = corners.mean(axis=0)
    theta_rad = np.radians(delta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    rotated = (corners - center) @ R.T + center
    return rotated.astype(np.float32)


def _expand_edges_if_undershooting(img, corners, n=9, max_expand_steps=1, step_px=2, patch_half=6, threshold_frac=0.10):
    """
    Complementary to _trim_uniform_frame (Tier 2's inward-only correction):
    checks for real board content just OUTSIDE each of the 4 edges, and if
    found, nudges that specific edge outward in small steps until no more
    real content is found beyond it.

    Confirmed necessary on a real photographed test image where Tier 1's
    homography fit was uniformly under-sized by a small but consistent
    ~3-4px on multiple independent edges (confirmed via direct pixel
    inspection: at a row/column confirmed to be inside real board content,
    the true light/dark transition midpoint sat ~3-4px further out than
    the fitted corner, on both the left and right edges independently).
    This is NOT a rotation/tilt error -- a small independent undershoot on
    each of several edges, on an already slightly-tilted board, creates a
    visual illusion of "needs more tilt" that a pure rotation correction
    cannot fix (confirmed: an earlier rotation-only correction attempt on
    this same image did not resolve the user's visual complaint).
    """
    corners = corners.astype(np.float32).copy()

    def sample_variance(quad, edge, offset_px):
        tl, tr, br, bl = quad
        if edge == 'top':
            p0, p1, direction = tl, tr, (tl - bl)
        elif edge == 'bottom':
            p0, p1, direction = bl, br, (bl - tl)
        elif edge == 'left':
            p0, p1, direction = tl, bl, (tl - tr)
        else:
            p0, p1, direction = tr, br, (tr - tl)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        means = []
        for i in range(n - 1):
            base = p0 + (p1 - p0) * (i + 0.5) / (n - 1) + direction * offset_px
            cx, cy = int(round(base[0])), int(round(base[1]))
            y0, y1 = max(0, cy - patch_half), min(img.shape[0], cy + patch_half + 1)
            x0, x1 = max(0, cx - patch_half), min(img.shape[1], cx + patch_half + 1)
            if y1 > y0 and x1 > x0:
                means.append(float(img[y0:y1, x0:x1].astype(np.float32).mean()))
        return float(np.var(means)) if len(means) >= 2 else 0.0

    tl, tr, br, bl = corners
    mid_top, mid_bot = tl + (tr - tl) * 0.5, bl + (br - bl) * 0.5
    # interior reference: mid-column band, varying by row
    means = []
    for i in range(n - 1):
        base = mid_top + (mid_bot - mid_top) * (i + 0.5) / (n - 1)
        cx, cy = int(round(base[0])), int(round(base[1]))
        y0, y1 = max(0, cy - patch_half), min(img.shape[0], cy + patch_half + 1)
        x0, x1 = max(0, cx - patch_half), min(img.shape[1], cx + patch_half + 1)
        if y1 > y0 and x1 > x0:
            means.append(float(img[y0:y1, x0:x1].astype(np.float32).mean()))
    interior_ref = float(np.var(means)) if len(means) >= 2 else 0.0
    threshold = max(30.0, interior_ref * threshold_frac)

    for edge in ['top', 'bottom', 'left', 'right']:
        for _ in range(max_expand_steps):
            # check a band just past the CURRENT edge position (small
            # positive offset from the edge itself, not from deep inside)
            if sample_variance(corners, edge, step_px / 2) < threshold:
                break
            tl, tr, br, bl = corners
            if edge == 'top':
                d = (tl - bl) / (np.linalg.norm(tl - bl) + 1e-6)
                corners = np.array([tl + d * step_px, tr + d * step_px, br, bl], dtype=np.float32)
            elif edge == 'bottom':
                d = (bl - tl) / (np.linalg.norm(bl - tl) + 1e-6)
                corners = np.array([tl, tr, br + d * step_px, bl + d * step_px], dtype=np.float32)
            elif edge == 'left':
                d = (tl - tr) / (np.linalg.norm(tl - tr) + 1e-6)
                corners = np.array([tl + d * step_px, tr, br, bl + d * step_px], dtype=np.float32)
            else:
                d = (tr - tl) / (np.linalg.norm(tr - tl) + 1e-6)
                corners = np.array([tl, tr + d * step_px, br + d * step_px, bl], dtype=np.float32)

    return corners


def find_board_grid_robust(img, n=9, hough_thresholds=(140, 120, 100, 80, 60), return_info=False):
    """
    Public entry point. Detect the 9x9 grid of board-square corner points
    for an (n-1)x(n-1) board (n=9 -> standard 8x8 chessboard).

    return_info: if True, returns (grid, tier_name, score) instead of just
        grid -- purely a diagnostic/QA aid (e.g. for a dataset-building
        script to log which tier resolved each image, so a systematic
        drift toward late tiers across a whole dataset is visible rather
        than silently accepted). Default False preserves the exact prior
        return type for existing callers.

    Four tiers, each a fallback for the previous -- only used if the prior
    tier produces zero valid candidates, never blended/compared across
    tiers by raw score (see _grid_alignment_score's docstring for why a
    single post-hoc pixel score isn't safe to compare across fundamentally
    different detector strategies):

      TIER 1 -- line-based homography (_find_grid_candidates_scaled). The
        primary method: Hough-detect grid lines, cluster into the board's
        two orientation families, robustly fit a 9-line index
        correspondence per family (RANSAC-style spacing/offset search),
        take every observed-line-pair intersection as one correspondence
        between an ideal uniform board coordinate and an image point, and
        fit a single homography (cv2.findHomography, RANSAC) through all of
        them -- the physically correct model for a flat board under any
        camera perspective, correctly reproducing the non-uniform line
        spacing a trapezoidal shot produces (naive evenly-spaced-line
        assumptions get this wrong at the extrapolated outer edges). Uses
        the most information (up to 81 point correspondences spanning the
        whole board) of the four tiers, and is validated across many test
        images including a low-contrast board where Tiers 2/3 both find
        literally nothing (no saturated color, no clean outer contour).
        Tried on the image as given AND after _autocrop_borders strips
        solid-color UI letterboxing, keeping whichever scores higher (see
        _autocrop_borders' docstring -- MAD-based flatness, robust to real
        camera/JPEG noise that inflates plain std on a genuinely flat bar).

      TIER 2 -- color/saturation quad (_find_quad_color,
        _find_quad_color_rect). Falls through to this only when Tier 1
        finds nothing (e.g. an image where piece artwork and on-board
        coordinate-label text produce as many spurious lines across all
        orientations as the true grid lines -- see PIPELINE_LOG.md's
        tough.jpg case, edge density 8.7% vs low single digits on clean
        renders). Thresholds only the saturated squares (most digital
        board themes use 1-2 distinctly colored square colors) into
        isolated per-square blobs -- no risky morphological closing needed
        -- and takes their convex hull; the hull's corners land on the
        board's true outer corners regardless of what surrounds it or how
        noisy the edges/lines are. Two variants (approxPolyDP-on-hull vs.
        minAreaRect) are both tried and validated by _quad_sanity_score.

      TIER 3 -- largest 4-point edge contour (_find_quad_edges). Last
        resort among the quad detectors: plain Canny + dilate + contour
        search for the biggest convex quadrilateral. Weakest signal of the
        three (most prone to grabbing a UI frame instead of the board --
        caught by _quad_sanity_score's border-hug check).

      TIER 4 -- axis-aligned bounding box from row/column mean brightness
        profiles (_find_grid_via_bbox). Only reached when Tiers 1-3 all
        find nothing -- the profile of a board that's mostly empty
        squares at very low contrast (as little as ~14 grey levels between
        adjacent squares) with almost no saturation and too few pieces to
        give Tiers 1-3 enough signal along several of the 9 grid lines.
        Averaging brightness across an entire row/column cancels per-pixel
        noise and reveals a clear step between "background outside the
        board" and "board interior" even when individual pixels barely
        differ -- critically, this needs NO signal from the interior 8x8
        structure at all, only the outer silhouette against a plain
        background. Trade-off: assumes the board is axis-aligned (a flat
        screenshot, not a perspective photo) with no correction for skew,
        which is why it's the last resort rather than tried first even
        though it can succeed with less signal than the others.

    If no tier produces a valid candidate, raises RuntimeError (see module
    docstring for the manual 4-corner fallback via grid_from_corners).

    Returns a (9, 9, 2) float32 array: grid[row, col] = (x, y) in the
    ORIGINAL (uncropped) image's coordinate system.
    """
    # ---- Tier 1: line-based homography ----
    all_candidates = []

    cands_full = _find_grid_candidates_scaled(img, n=n, hough_thresholds=hough_thresholds)
    for score, grid, info in cands_full:
        all_candidates.append((score, grid, dict(info, variant="original")))

    cropped, x_off, y_off = _autocrop_borders(img)
    if (x_off, y_off) != (0, 0) or cropped.shape != img.shape:
        cands_cropped = _find_grid_candidates_scaled(cropped, n=n, hough_thresholds=hough_thresholds)
        offset = np.array([x_off, y_off], dtype=np.float32)
        for score, grid, info in cands_cropped:
            all_candidates.append((score, grid + offset, dict(info, variant="autocropped")))

    all_candidates = [c for c in all_candidates if c[0] > 1e-6]
    if all_candidates:
        all_candidates.sort(key=lambda c: c[0], reverse=True)
        best = all_candidates[0]
        score, grid, info = best

        # NOTE: _expand_edges_if_undershooting and _refine_grid_tilt exist
        # but are deliberately NOT applied here by default. Both were
        # built and tested against a single real photographed test image
        # with a small (~3-4px per edge) systematic undershoot that
        # visually presented as a tilt problem. Three attempts (pure
        # rotation, aggressive expansion, conservative expansion) each
        # traded one problem for another: too little correction, too much
        # correction, or measurable ~2-3px drift on OTHER already-verified
        # -accurate images with no independent confirmation of whether
        # that drift was a hidden improvement or a regression. That
        # pattern -- not converging after three tuning attempts -- means
        # this correction is operating close to the noise floor of that
        # image's own blur, not a clean, confidently-generalizable fix.
        # Shipping it by default risked silently perturbing every other
        # Tier-1 image by a few px for uncertain benefit. Left in the
        # module for a specific caller to opt into and re-validate
        # against a broader test set, not wired into the default path.

        return (grid, "tier1_line_homography", score) if return_info else grid

    # ---- Tier 2: color/saturation quad ----
    tier2 = []
    for quad_fn, name in [(_find_quad_color, "color_hull"), (_find_quad_color_rect, "color_rect")]:
        for variant_img, x_off2, y_off2, variant_name in [(img, 0, 0, "original"),
                                                            (cropped, x_off, y_off, "autocropped")]:
            result = _find_grid_via_quad(variant_img, quad_fn, n=n)
            if result is None:
                continue
            score, grid, info = result
            offset = np.array([x_off2, y_off2], dtype=np.float32)
            tier2.append((score, grid + offset, dict(info, variant=f"{name}/{variant_name}")))

    if tier2:
        # within-tier tie-break: sanity score first, alignment as secondary
        tier2.sort(key=lambda c: (c[0], c[2]["alignment"]), reverse=True)
        best = tier2[0]
        return (best[1], "tier2_color_quad", best[0]) if return_info else best[1]

    # ---- Tier 3: largest edge-contour quad ----
    tier3 = []
    for variant_img, x_off3, y_off3, variant_name in [(img, 0, 0, "original"),
                                                        (cropped, x_off, y_off, "autocropped")]:
        result = _find_grid_via_quad(variant_img, _find_quad_edges, n=n)
        if result is None:
            continue
        score, grid, info = result
        offset = np.array([x_off3, y_off3], dtype=np.float32)
        tier3.append((score, grid + offset, dict(info, variant=f"edge_contour/{variant_name}")))

    if tier3:
        tier3.sort(key=lambda c: (c[0], c[2]["alignment"]), reverse=True)
        best = tier3[0]
        return (best[1], "tier3_edge_contour", best[0]) if return_info else best[1]

    # ---- Tier 4: axis-aligned bounding box from row/column brightness profiles ----
    tier4 = []
    for variant_img, x_off4, y_off4, variant_name in [(img, 0, 0, "original"),
                                                        (cropped, x_off, y_off, "autocropped")]:
        result = _find_grid_via_bbox(variant_img, n=n)
        if result is None:
            continue
        score, grid, info = result
        offset = np.array([x_off4, y_off4], dtype=np.float32)
        tier4.append((score, grid + offset, dict(info, variant=f"bbox/{variant_name}")))

    if tier4:
        tier4.sort(key=lambda c: (c[0], c[2]["alignment"]), reverse=True)
        best = tier4[0]
        return (best[1], "tier4_bbox", best[0]) if return_info else best[1]

    # ---- Tier 5: axis-aligned periodicity (edge-density autocorrelation + comb fit) ----
    tier5 = []
    for variant_img, x_off5, y_off5, variant_name in [(img, 0, 0, "original"),
                                                        (cropped, x_off, y_off, "autocropped")]:
        result = _find_grid_via_periodicity(variant_img, n=n)
        if result is None:
            continue
        score, grid, info = result
        offset = np.array([x_off5, y_off5], dtype=np.float32)
        tier5.append((score, grid + offset, dict(info, variant=f"periodicity/{variant_name}")))

    if tier5:
        tier5.sort(key=lambda c: (c[0], c[2]["alignment"]), reverse=True)
        best = tier5[0]
        return (best[1], "tier5_periodicity", best[0]) if return_info else best[1]

    raise RuntimeError(
        "Automatic grid detection failed across all five tiers (line-based "
        "homography, color/saturation quad, edge-contour quad, axis-aligned "
        "bounding box, axis-aligned periodicity). Fall back to manual "
        "4-corner selection via grid_from_corners() -- see module docstring."
    )


# Backwards-compatible alias.
find_board_grid = find_board_grid_robust


def grid_from_corners(corners, n=9):
    """
    Manual-fallback grid builder. Given the 4 outer corners of the board in
    image pixel coordinates -- ordered [top-left, top-right, bottom-right,
    bottom-left] -- returns the same (9, 9, 2) grid structure that
    find_board_grid_robust produces automatically, so slice_squares /
    make_debug_overlay / make_mosaic all work unchanged.

    This is the fallback for images where automatic line detection doesn't
    have enough signal to trust (see module docstring and PIPELINE_LOG.md
    for when that happens -- e.g. bold piece artwork + on-board coordinate
    labels + real camera noise all contributing spurious lines at a rate
    comparable to the true grid lines). Once you have 4 correct corners
    from any source (a human clicking them, a different detector, etc.) a
    single homography from an ideal uniform 9x9 board to those 4 corners is
    exact for a flat board under any camera perspective -- the same
    homography model find_board_grid_robust fits automatically, just given
    the corners directly instead of inferring them from line intersections.

    corners: array-like of 4 (x, y) points, [TL, TR, BR, BL].
    Returns a (9, 9, 2) float32 array: grid[row, col] = (x, y).
    """
    corners = np.array(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"expected 4 (x,y) corners [TL,TR,BR,BL], got shape {corners.shape}")

    board_corners = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(board_corners, corners)

    ideal = np.array([[j, i] for i in range(n) for j in range(n)], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(ideal, H).reshape(n, n, 2)
    return mapped.astype(np.float32)


def run_manual(image_path, out_dir, corners, cell_px=96):
    """
    Same as run(), but takes explicit board corners instead of running
    automatic detection -- the fallback path for images
    find_board_grid_robust can't confidently solve. See grid_from_corners.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    grid = grid_from_corners(corners)

    squares_dir = os.path.join(out_dir, "squares")
    os.makedirs(squares_dir, exist_ok=True)
    cells = slice_squares(img, grid, cell_px=cell_px)
    for (i, j), cell in cells.items():
        cv2.imwrite(os.path.join(squares_dir, f"row{i}_col{j}.png"), cell)

    cv2.imwrite(os.path.join(out_dir, "grid_overlay.png"), make_debug_overlay(img, grid))
    cv2.imwrite(os.path.join(out_dir, "mosaic.png"), make_mosaic(cells, cell_px=cell_px))
    np.save(os.path.join(out_dir, "grid_points.npy"), grid)

    print(f"Wrote 64 squares to {squares_dir}")
    print(f"Wrote debug overlay to {os.path.join(out_dir, 'grid_overlay.png')}")
    print(f"Wrote contact sheet to {os.path.join(out_dir, 'mosaic.png')}")
    return grid, cells


def slice_squares(img, grid, cell_px=96):
    """
    Given the (9,9,2) grid of corner points, warp each of the 64 cells to
    its own `cell_px` x `cell_px` square using a per-cell perspective
    transform (robust to trapezoidal skew across the whole board).

    Returns a dict {(row, col): cell_image} for row,col in 0..7.
    """
    dst = np.array([[0, 0], [cell_px, 0], [cell_px, cell_px], [0, cell_px]], dtype=np.float32)
    cells = {}
    for i in range(8):
        for j in range(8):
            tl, tr = grid[i, j], grid[i, j + 1]
            br, bl = grid[i + 1, j + 1], grid[i + 1, j]
            src = np.array([tl, tr, br, bl], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            cells[(i, j)] = cv2.warpPerspective(img, M, (cell_px, cell_px))
    return cells


def make_debug_overlay(img, grid):
    vis = img.copy()
    for i in range(9):
        cv2.polylines(vis, [grid[i, :, :].astype(int)], False, (0, 255, 0), 1)
    for j in range(9):
        cv2.polylines(vis, [grid[:, j, :].astype(int)], False, (0, 255, 0), 1)
    for i in range(9):
        for j in range(9):
            x, y = grid[i, j]
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    return vis


def make_mosaic(cells, cell_px=96, pad=2):
    mosaic = np.full((8 * (cell_px + pad) + pad, 8 * (cell_px + pad) + pad, 3), 255, dtype=np.uint8)
    for (i, j), cell in cells.items():
        y0 = pad + i * (cell_px + pad)
        x0 = pad + j * (cell_px + pad)
        mosaic[y0:y0 + cell_px, x0:x0 + cell_px] = cell
    return mosaic


def run(image_path, out_dir, cell_px=96):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    grid = find_board_grid_robust(img)

    squares_dir = os.path.join(out_dir, "squares")
    os.makedirs(squares_dir, exist_ok=True)
    cells = slice_squares(img, grid, cell_px=cell_px)
    for (i, j), cell in cells.items():
        cv2.imwrite(os.path.join(squares_dir, f"row{i}_col{j}.png"), cell)

    cv2.imwrite(os.path.join(out_dir, "grid_overlay.png"), make_debug_overlay(img, grid))
    cv2.imwrite(os.path.join(out_dir, "mosaic.png"), make_mosaic(cells, cell_px=cell_px))
    np.save(os.path.join(out_dir, "grid_points.npy"), grid)

    print(f"Wrote 64 squares to {squares_dir}")
    print(f"Wrote debug overlay to {os.path.join(out_dir, 'grid_overlay.png')}")
    print(f"Wrote contact sheet to {os.path.join(out_dir, 'mosaic.png')}")
    return grid, cells


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chessboard_grid_segmentation.py <image_path> [output_dir] [--corners x1,y1,x2,y2,x3,y3,x4,y4]")
        print("  --corners takes 4 points in image pixel coords, ordered TL,TR,BR,BL --")
        print("  use this fallback when automatic detection raises RuntimeError.")
        sys.exit(1)
    image_path = sys.argv[1]
    args = sys.argv[2:]
    corners_arg = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--corners":
            nums = [float(x) for x in args[i + 1].split(",")]
            if len(nums) != 8:
                print("--corners needs exactly 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL)")
                sys.exit(1)
            corners_arg = [[nums[k], nums[k + 1]] for k in range(0, 8, 2)]
            i += 2
        else:
            positional.append(args[i])
            i += 1
    out_dir = positional[0] if positional else "chessboard_grid_output"
    os.makedirs(out_dir, exist_ok=True)
    if corners_arg is not None:
        run_manual(image_path, out_dir, corners_arg)
    else:
        run(image_path, out_dir)
