# Let-It-Chessify — Backend

FastAPI service that turns a photo of a chessboard (digital/screenshot **or**
physical board) into a FEN string, plus a Stockfish-powered analysis endpoint.
This directory is the backend subrepo of the larger **Let-It-Chessify**
project — the actual launcher (`run.py`) and project root live one level
above this folder.

## What it does

The pipeline runs in three confirmed steps, shared by both board types:

```
/detect-corners  →  /refine-corners (optional)  →  /classify
```

1. **`/detect-corners`** — locates the board in the uploaded image and
   returns a 9×9 grid of intersection points plus a debug overlay so the
   frontend can show the detected grid to the user for confirmation.
2. **`/refine-corners`** — recomputes the grid after the user manually drags
   a corner (debounced on the frontend).
3. **`/classify`** — runs piece recognition on the confirmed grid and
   returns a FEN string, per-square labels, and confidence scores.

Two independent detection pipelines are used depending on the image type:

|                   | Digital (screenshot / book photo)                                                                                                                       | Physical (photo of a real board)                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Board detection   | Classical CV — Hough lines, contour/color/edge quads, periodicity analysis, tilt correction (`chessboard_grid_segmentation.py`, `classical_cv.py`) | YOLOv8n-pose corner keypoint model (`corner_detector.py`)                                                                                                 |
| Piece detection   | Implicit — one crop per grid cell                                                                                                                      | YOLOv8 single-class piece detector on the raw image (`piece_detector.py`)                                                                                 |
| Square assignment | Fixed grid cell → square                                                                                                                               | Homography (`compute_homography` in `board_mapper.py` maps raw image coords → normalized 8×8 board space), highest-confidence piece wins on conflicts |
| Classifier        | ONNX CNN, 13 classes (`Empty` + 6 white + 6 black)                                                                                                    | ONNX CNN, 12 classes (no`Empty` class — a square is labeled `Empty` when no piece was detected on it at all, not via a confidence cutoff)              |
| Preprocessing     | Grayscale → background normalization → CLAHE                                                                                                          | CLAHE on HSV V-channel → grayscale (order matters — see`preprocessing.py`)                                                                              |

Both classifiers are loaded together at startup (`classifier.py`,
`DualClassifier`) and selected per-request via `is_physical`.

A separate `/analyze` endpoint wraps a local **Stockfish** binary
(`engine.py`) to return an evaluation score, mate detection, and top
candidate moves for a given FEN.

## API endpoints

| Method | Path                     | Purpose                                                   |
| ------ | ------------------------ | --------------------------------------------------------- |
| GET    | `/health`              | Model/engine load status                                  |
| POST   | `/detect-corners`      | Step 1 — detect board grid from an uploaded image        |
| POST   | `/refine-corners`      | Step 2 — recompute grid from user-adjusted corners       |
| POST   | `/classify`            | Step 3 — classify pieces on a confirmed grid → FEN      |
| POST   | `/detect`              | Legacy one-shot detect + classify                         |
| POST   | `/detect-with-corners` | Legacy manual-corner classify (digital only)              |
| POST   | `/edit-fen`            | Rebuild a FEN from a manually edited set of square labels |
| POST   | `/analyze`             | Stockfish evaluation + best moves for a FEN               |

The frontend contract is the 3-step flow (`/detect-corners` →
`/refine-corners` → `/classify`); the other endpoints exist for legacy
compatibility.

## Project layout

```
backend/
├── main.py                 # FastAPI app, endpoints, pipeline orchestration
├── engine.py                # Stockfish UCI wrapper (eval, top moves)
├── requirements.txt
├── __init__.py               # marks backend/ as a package (for `backend.*` imports)
├── models/
│   └── README.md            # Stockfish binary download instructions
└── detection/
    ├── __init__.py                       # marks detection/ as a package
    ├── chessboard_grid_segmentation.py  # Classical CV board/grid detection (digital)
    ├── classical_cv.py                  # Board contour/color/edge detection helpers
    ├── corner_detector.py               # YOLOv8n-pose corner detection (physical)
    ├── piece_detector.py                # YOLOv8 piece bbox detection (physical)
    ├── board_mapper.py                  # Homography + square assignment (physical)
    ├── classifier.py                    # Dual ONNX CNN classifier (synthetic + physical)
    ├── preprocessing.py                 # Two separate, faithful preprocessing recipes
    └── fen.py                           # Square-label dict → FEN string
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Models required at runtime

These are **not** bundled in the repo and must be present (paths are
overridable via environment variables — see below):

- `runs/synthetic/square_classifier.onnx` + `label_map.json` — digital CNN
- `runs/physical/square_classifier.onnx` + `label_map.json` — physical CNN
- `models/corner_detection/train-2/weights/best.pt` — YOLOv8n-pose corners
- `models/piece_detection/train/weights/best.pt` — YOLOv8 piece detector
- `backend/models/stockfish` (or `stockfish.exe`) — see
  [`backend/models/README.md`](models/README.md) for the download link

`GET /health` reports which of these were found at startup.

### Environment variables (optional overrides)

```
SYNTHETIC_MODEL_PATH
SYNTHETIC_LABEL_MAP_PATH
PHYSICAL_MODEL_PATH
PHYSICAL_LABEL_MAP_PATH
CORNER_MODEL_PATH
PIECE_DETECTION_MODEL_PATH
```

## Running

This backend is a subdirectory of the full **Let-It-Chessify** repo, and is
started from a launcher at the **project root** — not with `uvicorn`
directly:

```bash
python run.py                    # http://localhost:8000, host 0.0.0.0
python run.py --port 8001        # custom port
python run.py --no-reload        # currently a no-op — see note below
```

`run.py` exists specifically to get path setup right across OSes: it
inserts the project root into `sys.path` and sets `PYTHONPATH` *before*
uvicorn spawns any subprocess, so the `backend.*` absolute imports used
throughout this subrepo resolve correctly no matter where you invoke it
from. Swagger UI is printed on startup at `http://localhost:<port>/docs`.

> **Note:** `--no-reload` is parsed but not currently wired up —
> `uvicorn.run()` is called with `reload=False` unconditionally, so
> auto-reload is always off regardless of the flag. If you want reload
> during development, run `uvicorn backend.main:app --reload` from the
> project root manually instead, or fix `run.py` to pass
> `reload=not args.no_reload`.

CORS is currently open to `localhost:5173`, `localhost:3000`, and their
`127.0.0.1` equivalents for local frontend development.

## Notes for contributors

- The digital and physical preprocessing recipes must stay separate —
  see the header comment in `preprocessing.py`; merging them previously
  caused an accuracy regression.
- Board orientation is not auto-detected; the frontend's "Flip board"
  control handles it.
- Piece-crop padding for the physical pipeline is intentionally `0`
  (`PIECE_CROP_PADDING` in `main.py`) to match the tight-bbox crops the
  physical CNN was trained on.
- On Physical Board `Empty` squares are decided by the piece **detector**, not the
  classifier: any square with no assigned piece bounding box is `Empty`.
