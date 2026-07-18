"""
Chess OCR — FastAPI Backend v2.2 (New Physical Pipeline)
==========================================================
Digital pipeline  (unchanged):
    /detect-corners → /refine-corners → /classify
    find_board_grid_robust → per-cell warp → preprocess_synthetic → CNN (13cls)

Physical pipeline (new — Section 21 rework):
    /detect-corners → /refine-corners → /classify
    YOLOv8 corners → homography → YOLOv8 piece detection on RAW image
    → per-piece CNN classification → square assignment via homography
    → FEN

Frontend contract: UNCHANGED — same 3 endpoints, same response shape.
All pipeline changes are internal to /classify.

Key decisions logged:
    - No padding on piece crops (matches CNN training distribution,
      which used tight bboxes) — PIECE_CROP_PADDING = 0
    - Conflict resolution: highest detector confidence wins per square
    - Empty squares: any square with no detected piece center = Empty
      (50% threshold from Section 21.3 no longer needed for physical)
    - Board orientation: not auto-detected, user uses Flip board button
"""

import sys
import os
import asyncio
import threading
import base64
import traceback
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_backend_dir = Path(__file__).parent
_project_dir = _backend_dir.parent
for p in (_project_dir, _backend_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.detection.chessboard_grid_segmentation import (
    find_board_grid_robust,
    grid_from_corners,
    make_debug_overlay,
    slice_squares as seg_slice_squares,
    _refine_quad_via_edge_peaks,
)
from backend.detection.classical_cv         import order_corners
from backend.detection.corner_detector      import detect_physical_board_corners
from backend.detection.piece_detector       import detect_pieces
from backend.detection.board_mapper         import assign_pieces_to_squares
from backend.detection.preprocessing        import (
    preprocess_synthetic_crop,
    preprocess_physical_crop,
    resize_to_model_input,
)
from backend.detection.classifier           import DualClassifier
from backend.detection.fen                  import labels_to_fen
from backend.engine                         import StockfishEngine

import chess

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = _project_dir

SYNTHETIC_MODEL_PATH     = Path(os.environ.get("SYNTHETIC_MODEL_PATH",
    PROJECT_ROOT / "runs" / "synthetic" / "square_classifier.onnx"))
SYNTHETIC_LABEL_MAP_PATH = Path(os.environ.get("SYNTHETIC_LABEL_MAP_PATH",
    PROJECT_ROOT / "runs" / "synthetic" / "label_map.json"))
PHYSICAL_MODEL_PATH      = Path(os.environ.get("PHYSICAL_MODEL_PATH",
    PROJECT_ROOT / "runs" / "physical" / "square_classifier.onnx"))
PHYSICAL_LABEL_MAP_PATH  = Path(os.environ.get("PHYSICAL_LABEL_MAP_PATH",
    PROJECT_ROOT / "runs" / "physical" / "label_map.json"))
CORNER_MODEL_PATH        = Path(os.environ.get("CORNER_MODEL_PATH",
    PROJECT_ROOT / "models" / "corner_detection" / "train-2" / "weights" / "best.pt"))
PIECE_DETECTION_MODEL_PATH = Path(os.environ.get("PIECE_DETECTION_MODEL_PATH",
    PROJECT_ROOT / "models" / "piece_detection" / "train" / "weights" / "best.pt"))

CORNER_DETECTION_CONF_THRESHOLD = 0.5
PIECE_DETECTION_CONF_THRESHOLD  = 0.25
CELL_PX = 96

# Padding added to each side of the YOLO piece bbox before cropping and
# resizing to 64x64. The physical CNN was trained on TIGHT bboxes with no
# padding (see preprocessing.py), so this must stay 0 -- any padding here
# shifts the crop's scale/framing away from what the classifier was trained
# on, which hurts accuracy rather than helping it. Kept as a named constant
# (rather than inlining 0) so the bounds-clamping below still reads clearly.
PIECE_CROP_PADDING = 0

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Chess OCR API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000",
                   "http://127.0.0.1:5173","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

classifier = DualClassifier(
    synthetic_model_path     = SYNTHETIC_MODEL_PATH,
    synthetic_label_map_path = SYNTHETIC_LABEL_MAP_PATH,
    physical_model_path      = PHYSICAL_MODEL_PATH,
    physical_label_map_path  = PHYSICAL_LABEL_MAP_PATH,
)
engine = None

@app.on_event("startup")
async def startup():
    global engine
    print(f"[Startup] Synthetic :    {SYNTHETIC_MODEL_PATH}  exists={SYNTHETIC_MODEL_PATH.exists()}")
    print(f"[Startup] Physical  :    {PHYSICAL_MODEL_PATH}  exists={PHYSICAL_MODEL_PATH.exists()}")
    print(f"[Startup] Corner    :    {CORNER_MODEL_PATH}  exists={CORNER_MODEL_PATH.exists()}")
    print(f"[Startup] PieceDet  :    {PIECE_DETECTION_MODEL_PATH}  exists={PIECE_DETECTION_MODEL_PATH.exists()}")
    try:
        engine = StockfishEngine(depth=40, move_time=12)
        print(f"[Startup] Stockfish:     {engine.path}")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")

# ── Pydantic models ───────────────────────────────────────────────────────────

class RefineCornersRequest(BaseModel):
    image_b64:   str
    corners:     list
    is_physical: bool = False

class ClassifyRequest(BaseModel):
    image_b64:   str
    grid:        list   # (9,9,2) from /detect-corners — used for digital
                        # and for deriving the 4 outer corners for physical
    is_physical: bool = False

class CornersRequest(BaseModel):
    image_b64: str
    corners:   list

class EditFenRequest(BaseModel):
    square_labels: dict
    turn:          str = "w"

class AnalyzeRequest(BaseModel):
    fen:       str
    turn:      str = "w"
    num_moves: int = 3

# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image_bytes(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return np.array(img)

def load_image_b64(b64_str):
    raw     = base64.b64decode(b64_str)
    arr     = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, img_bgr

def rgb_to_b64(img_rgb):
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def bgr_to_b64(img_bgr):
    return rgb_to_b64(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def grid_to_list(g): return g.tolist()
def list_to_grid(l): return np.array(l, dtype=np.float32)

def corners_from_grid(grid):
    """Extract [TL, TR, BR, BL] outer corners from (9,9,2) grid."""
    return np.array([grid[0,0], grid[0,8], grid[8,8], grid[8,0]], dtype=np.float32)

# ── Grid helpers ──────────────────────────────────────────────────────────────

def refine_grid(img_bgr, corners_np, is_physical=False):
    """
    Section 25.2: grid_from_corners + _refine_quad_via_edge_peaks.
    Outer corners always honored exactly; interior lines snap to actual boundaries.

    IMPORTANT: _refine_quad_via_edge_peaks assumes an axis-aligned (or
    near-axis-aligned) quad -- see its docstring. A physical board photo is
    taken at an arbitrary camera angle, so that assumption doesn't hold:
    running it there silently replaces the true perspective quad (and any
    corner position a user just dragged into place) with an axis-aligned
    rectangle, which is wrong and makes manually placed/dragged corners
    appear to "snap back" to the wrong spot. For physical boards we skip
    that pass entirely and trust the corners exactly as given, using a
    full projective homography (grid_from_corners) to build the grid.
    """
    if is_physical:
        return grid_from_corners(corners_np, n=9)

    grid = grid_from_corners(corners_np, n=9)
    quad = [corners_np[0].tolist(), corners_np[1].tolist(),
            corners_np[2].tolist(), corners_np[3].tolist()]
    try:
        refined = _refine_quad_via_edge_peaks(img_bgr, quad, n=9, search_radius=15)
        grid    = grid_from_corners(np.array(refined, dtype=np.float32), n=9)
    except Exception:
        pass
    return grid

# ── Digital classification (unchanged) ───────────────────────────────────────

def classify_digital(img_bgr, grid_np):
    """
    Per-cell perspective warp → preprocess_synthetic_crop → Synthetic CNN (13cls).
    find_board_grid_robust already handles all the hard detection work.
    """
    cells  = seg_slice_squares(img_bgr, grid_np, cell_px=CELL_PX)
    crops  = {}
    for (row, col), cell_bgr in cells.items():
        pil      = Image.fromarray(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB))
        rank     = 7 - row
        file_idx = col
        sq_idx   = chess.square(file_idx, rank)
        processed = preprocess_synthetic_crop(pil, file_idx, rank)
        crops[sq_idx] = resize_to_model_input(processed, size=64)

    return classifier.classify_batch(crops, is_physical=False)

# ── Physical classification (new pipeline) ────────────────────────────────────

def classify_physical(img_rgb, img_bgr, corners_np):
    """
    New physical pipeline (replaces 1.5x crop approach):

    1. Run piece detection YOLO on the ORIGINAL raw image
       → N bounding boxes (single class "piece") + confidence scores
    2. For each bbox:
       a. Crop the tight bbox from the original image, no padding
          (matches the CNN's training distribution — see preprocessing.py)
       b. Apply preprocess_physical_crop (CLAHE on HSV V-channel — matches training)
       c. Resize to 64x64
       d. Run through physical CNN classifier → piece label (12 classes)
    3. Transform each bbox center via homography H (corners → 8x8 space)
       → assign to one of 64 squares (Option B: direct 8x8 mapping)
    4. Conflict resolution: highest detector confidence wins per square
    5. Any square with no detected piece = Empty (no threshold needed)
    6. Build full 64-square label dict → FEN

    Returns: results dict {sq_idx → {"label": str, "confidence": float, ...}}
    """
    h_img, w_img = img_rgb.shape[:2]

    # Step 1 — piece detection on raw image
    pieces = detect_pieces(
        img_rgb,
        PIECE_DETECTION_MODEL_PATH,
        conf_threshold=PIECE_DETECTION_CONF_THRESHOLD,
    )

    if not pieces:
        # No pieces detected — return all-empty board
        results = {}
        for sq in chess.SQUARES:
            results[sq] = {"label": "Empty", "confidence": 1.0,
                           "detection_confidence": 0.0}
        return results

    # Step 2 — classify each detected piece
    piece_labels = []
    for piece in pieces:
        xmin, ymin, xmax, ymax = piece["bbox"]

        # Tight bbox, clamped to image bounds (no padding -- PIECE_CROP_PADDING
        # is 0 to match the tight-bbox crops the classifier was trained on).
        xmin_p = max(0,     xmin - PIECE_CROP_PADDING)
        ymin_p = max(0,     ymin - PIECE_CROP_PADDING)
        xmax_p = min(w_img, xmax + PIECE_CROP_PADDING)
        ymax_p = min(h_img, ymax + PIECE_CROP_PADDING)

        # Crop from original RGB image
        crop_rgb = img_rgb[int(ymin_p):int(ymax_p), int(xmin_p):int(xmax_p)]
        if crop_rgb.size == 0:
            continue

        pil_crop  = Image.fromarray(crop_rgb)
        processed = preprocess_physical_crop(pil_crop)
        processed = resize_to_model_input(processed, size=64)

        # Classify single crop
        single_result = classifier.classify_batch(
            {0: processed}, is_physical=True
        )
        label      = single_result[0]["label"]
        cnn_conf   = single_result[0]["confidence"]

        piece_labels.append({
            **piece,               # bbox, center, confidence (detector)
            "label":               label,
            "cnn_confidence":      cnn_conf,
            "detector_confidence": piece["confidence"],
        })

    # Step 3 — assign pieces to squares via homography
    # assign_pieces_to_squares uses detector confidence for conflict resolution
    square_assignments = assign_pieces_to_squares(piece_labels, corners_np)

    # Step 4 — build full 64-square result dict (Empty for unoccupied squares)
    results = {}
    for sq in chess.SQUARES:
        if sq in square_assignments:
            assignment = square_assignments[sq]
            results[sq] = {
                "label":               assignment["label"],
                "confidence":          assignment["cnn_confidence"],
                "detector_confidence": assignment["detector_confidence"],
                "board_xy":            assignment.get("board_xy"),
                "sq_name":             chess.square_name(sq),
            }
        else:
            results[sq] = {
                "label":      "Empty",
                "confidence": 1.0,    # no piece detected = confident Empty
                "detector_confidence": 0.0,
            }

    return results

# ── Unified classify dispatcher ───────────────────────────────────────────────

def classify_from_grid(img_rgb, img_bgr, grid_np, is_physical):
    """Route to the correct pipeline based on is_physical."""
    if is_physical:
        corners_np = corners_from_grid(grid_np)
        return classify_physical(img_rgb, img_bgr, corners_np)
    else:
        return classify_digital(img_bgr, grid_np)

def build_classify_response(results, is_physical, corner_confidence=None):
    labels        = {sq: r["label"] for sq, r in results.items()}
    fen, warnings = labels_to_fen(labels, turn="w")

    # For digital: flag low CNN confidence. For physical: flag low detector confidence.
    low_thr  = 0.75
    low_conf = []
    for sq, r in results.items():
        conf = r["confidence"]
        if is_physical:
            # Also surface detector confidence for physical
            det_conf = r.get("detector_confidence", 1.0)
            if r["label"] != "Empty" and (conf < low_thr or det_conf < 0.5):
                low_conf.append({
                    "square":               chess.square_name(sq),
                    "label":                r["label"],
                    "cnn_confidence":       round(conf, 3),
                    "detector_confidence":  round(det_conf, 3),
                })
        else:
            if conf < low_thr:
                low_conf.append({
                    "square":     chess.square_name(sq),
                    "label":      r["label"],
                    "confidence": round(conf, 3),
                })

    resp = {
        "success":                True,
        "is_physical":             is_physical,
        "fen":                     fen,
        "warnings":                warnings,
        "square_labels":           {chess.square_name(k): v for k, v in labels.items()},
        "confidences":             {chess.square_name(k): round(v["confidence"], 3)
                                    for k, v in results.items()},
        "low_confidence_squares":  low_conf,
    }
    if is_physical and corner_confidence is not None:
        resp["corner_detection_confidence"] = round(corner_confidence, 3)
    return resp

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "2.2.0",
        "synthetic_classifier_loaded":  classifier.is_loaded(is_physical=False),
        "physical_classifier_loaded":   classifier.is_loaded(is_physical=True),
        "stockfish_loaded":             engine is not None,
        "model_files_exist": {
            "synthetic_onnx":    SYNTHETIC_MODEL_PATH.exists(),
            "physical_onnx":     PHYSICAL_MODEL_PATH.exists(),
            "corner_pt":         CORNER_MODEL_PATH.exists(),
            "piece_detection_pt": PIECE_DETECTION_MODEL_PATH.exists(),
        },
        "piece_crop_padding": PIECE_CROP_PADDING,
        "stockfish_path":     engine.path if engine else None,
    }


@app.post("/detect-corners")
async def detect_corners(file: UploadFile = File(...), is_physical: bool = Form(False)):
    """
    STEP 1 — detection only. Returns corners + 9x9 grid + overlay.
    Frontend shows this for confirmation before /classify.
    """
    try:
        img_rgb = load_image_bytes(await file.read())
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        corner_confidence = None

        if is_physical:
            corners_np, corner_confidence = detect_physical_board_corners(
                img_rgb, CORNER_MODEL_PATH,
                conf_threshold=CORNER_DETECTION_CONF_THRESHOLD,
            )
            if corners_np is None:
                return {
                    "success": False, "needs_reupload": True,
                    "message": "Could not detect board corners. "
                               "Retake photo with full board visible and well-lit.",
                    "original_image_b64": rgb_to_b64(img_rgb),
                }
            grid      = refine_grid(img_bgr, corners_np, is_physical=True)
            tier_used = "yolov8n-pose"

        else:
            try:
                grid, tier_used, score = find_board_grid_robust(img_bgr, return_info=True)
            except RuntimeError:
                return {
                    "success": False, "needs_manual": True,
                    "message": "Could not detect chessboard automatically.",
                    "original_image_b64": rgb_to_b64(img_rgb),
                }
            corner_confidence = float(score)

        overlay    = make_debug_overlay(img_bgr, grid)
        corners_np = corners_from_grid(grid)

        return {
            "success":            True,
            "is_physical":         is_physical,
            "corners":             corners_np.tolist(),
            "grid":                grid_to_list(grid),
            "overlay_image_b64":   bgr_to_b64(overlay),
            "original_image_b64":  rgb_to_b64(img_rgb),
            "tier_used":           tier_used,
            "corner_confidence":   round(corner_confidence, 3) if corner_confidence else None,
            "message": "Grid detected. Confirm or drag corners to adjust.",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine-corners")
async def refine_corners_endpoint(req: RefineCornersRequest):
    """
    STEP 2 (optional, on corner drag) — recompute grid.
    Debounce ~150ms in frontend.

    This always fires in response to the user manually dragging a corner --
    for BOTH digital and physical boards. A "digital" upload can still be a
    photo of a book/screen taken at an angle (real perspective distortion,
    not just an axis-aligned scan), so it gets exactly the same treatment as
    a physical board here: the corners the user placed are honored exactly,
    via a full projective homography (grid_from_corners), with no
    axis-aligned "edge peak" snapping. That snapping step is only
    appropriate for the initial *automatic* detection pass (see
    /detect-corners), not for a position the user just deliberately set.
    """
    try:
        img_rgb, img_bgr = load_image_b64(req.image_b64)
        corners_np = np.array(req.corners, dtype=np.float32)
        if corners_np.shape != (4, 2):
            raise HTTPException(status_code=422, detail="corners must be [[x,y]] x 4")
        corners_np = order_corners(corners_np)
        grid       = grid_from_corners(corners_np, n=9)
        overlay    = make_debug_overlay(img_bgr, grid)
        return {
            "success":           True,
            "corners":           corners_from_grid(grid).tolist(),
            "grid":              grid_to_list(grid),
            "overlay_image_b64": bgr_to_b64(overlay),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest):
    """
    STEP 3 — classification on confirmed grid → FEN.

    Digital:  per-cell warp → preprocess_synthetic → Synthetic CNN (13cls)
    Physical: YOLO piece detect on raw image → crop each piece →
              preprocess_physical → CNN (12cls) → homography → square assign
    """
    try:
        img_rgb, img_bgr = load_image_b64(req.image_b64)
        grid_np          = list_to_grid(req.grid)
        if grid_np.shape != (9, 9, 2):
            raise HTTPException(status_code=422, detail="grid must be shape (9,9,2)")

        results = classify_from_grid(img_rgb, img_bgr, grid_np, req.is_physical)
        return build_classify_response(results, req.is_physical)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect(file: UploadFile = File(...), is_physical: bool = Form(False)):
    """Legacy one-shot. New frontend uses /detect-corners → /classify."""
    try:
        img_rgb = load_image_bytes(await file.read())
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        corner_confidence = None

        if is_physical:
            corners_np, corner_confidence = detect_physical_board_corners(
                img_rgb, CORNER_MODEL_PATH,
                conf_threshold=CORNER_DETECTION_CONF_THRESHOLD,
            )
            if corners_np is None:
                return {"success": False, "needs_reupload": True,
                        "message": "Could not detect board corners.",
                        "original_image_b64": rgb_to_b64(img_rgb)}
            grid = refine_grid(img_bgr, corners_np, is_physical=True)
        else:
            grid, _, score = find_board_grid_robust(img_bgr, return_info=True)
            corner_confidence = float(score)
            if grid is None:
                raise HTTPException(status_code=422, detail="Could not detect chessboard.")

        results    = classify_from_grid(img_rgb, img_bgr, grid, is_physical)
        overlay    = make_debug_overlay(img_bgr, grid)
        resp       = build_classify_response(results, is_physical, corner_confidence)
        resp["warped_board_b64"] = bgr_to_b64(overlay)
        resp["corners"]          = corners_from_grid(grid).tolist()
        resp["grid"]             = grid_to_list(grid)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-with-corners")
async def detect_with_corners(req: CornersRequest):
    """Legacy manual-corner (digital only)."""
    try:
        img_rgb, img_bgr = load_image_b64(req.image_b64)
        corners_np = order_corners(np.array(req.corners, dtype=np.float32))
        grid       = refine_grid(img_bgr, corners_np)
        results    = classify_from_grid(img_rgb, img_bgr, grid, is_physical=False)
        overlay    = make_debug_overlay(img_bgr, grid)
        resp       = build_classify_response(results, is_physical=False)
        resp["warped_board_b64"] = bgr_to_b64(overlay)
        resp["grid"]             = grid_to_list(grid)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit-fen")
async def edit_fen(req: EditFenRequest):
    try:
        labels = {
            chess.parse_square(k) if isinstance(k, str) and len(k) == 2 else int(k): v
            for k, v in req.square_labels.items()
        }
        fen, warnings = labels_to_fen(labels, turn=req.turn)
        return {"success": True, "fen": fen, "warnings": warnings}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Stockfish not available.")
    try:
        chess.Board(req.fen)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid FEN string")
    try:
        result = engine.analyze(fen=req.fen, turn=req.turn, num_moves=min(req.num_moves, 5))
        return {"success": True, **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """
    Live version of /analyze: streams a fresh eval snapshot after every
    depth Stockfish reports, so the client can show depth counting up
    (1/25, 2/25, ...) with the eval bar / best moves refining in real time,
    instead of waiting for one final result.

    Client sends one JSON message to kick things off:
        {"fen": "...", "turn": "w", "num_moves": 3}
    Server streams back:
        {"success": true, "depth": N, "done": false, "eval_cp": ..., ...}
        ... one message per depth update ...
        {"success": true, "depth": FINAL, "done": true, ...}
    or on error:
        {"success": false, "error": "..."}
    """
    await websocket.accept()

    if engine is None:
        await websocket.send_json({"success": False, "error": "Stockfish not available."})
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    fen       = data.get("fen")
    turn      = data.get("turn", "w")
    num_moves = min(int(data.get("num_moves", 3)), 5)

    try:
        chess.Board(fen)
    except Exception:
        await websocket.send_json({"success": False, "error": "Invalid FEN string"})
        await websocket.close()
        return

    loop         = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE         = object()  # sentinel meaning "generator finished, stop reading"
    cancel_event = threading.Event()

    def run_stream():
        try:
            for update in engine.analyze_stream(
                fen=fen, turn=turn, num_moves=num_moves, cancel_event=cancel_event
            ):
                if cancel_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(update), loop)
        except Exception as e:
            if not cancel_event.is_set():
                asyncio.run_coroutine_threadsafe(
                    queue.put({"success": False, "error": str(e)}), loop
                )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(DONE), loop)

    threading.Thread(target=run_stream, daemon=True).start()

    async def watch_disconnect():
        # Blocks on receive() until the client closes its end. This runs
        # concurrently with the queue-reading loop below so a disconnect is
        # noticed immediately -- not just when we happen to next try to
        # send something (which, without this, would let Stockfish keep
        # calculating on an abandoned position in the background).
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        cancel_event.set()

    disconnect_task = asyncio.create_task(watch_disconnect())

    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {get_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if disconnect_task in done:
                get_task.cancel()
                break

            update = get_task.result()
            if update is DONE:
                break
            if isinstance(update, dict) and update.get("success") is False:
                await websocket.send_json(update)
                break
            await websocket.send_json({"success": True, **update})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"success": False, "error": str(e)})
        except Exception:
            pass
    finally:
        cancel_event.set()
        if not disconnect_task.done():
            disconnect_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
