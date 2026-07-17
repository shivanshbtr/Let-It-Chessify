# ♛ Let It Chessify

**Photograph a chessboard — or screenshot one — and get an interactive analysis in seconds.**

Let It Chessify is a fully offline chess board recognition and analysis tool. Point your camera at a physical board (or paste a screenshot from Lichess / Chess.com), and the app detects the position, lets you verify and fix the result, then hands it off to Stockfish for evaluation — with a live score bar, coloured move arrows, and a playable analysis board.

---

## Features

- **Two-pipeline architecture** — separate detection paths for digital screenshots and physical board photos, each optimised for its source
- **5-tier digital board detection** — robust cascade from Hough-line homography down to FFT autocorrelation, handles Lichess, Chess.com, and any standard board theme
- **YOLO corner detection for physical boards** — trained YOLOv8n-pose model (mAP50-95: 0.995) precisely locates the four board corners from real photos
- **Interactive grid confirmation** — draggable corner handles let you nudge the detected grid before classification, covering any remaining edge cases
- **Physical piece detection + classification** — a single-class YOLO piece detector finds every piece in the raw photo; a lightweight CNN (0.08 MB ONNX) classifies each one into 12 piece types
- **Digital square classifier** — a separate CNN (0.08 MB ONNX) trained on 35 Chess.com themes + board-position rendering, handles every major digital board style
- **Board editor** — drag pieces between squares, right-click to remove, click a palette piece to add; flip the board 180° if the camera was from the wrong side
- **Stockfish analysis** — score bar mode (play moves freely, bar updates after each move) and move suggestion mode (top 3 moves as green / blue / yellow arrows with eval lines)
- **PGN import** — paste a PGN string and browse the game move by move in the analysis board
- **Fully offline** — all models run locally; no images or moves are sent to any server

---

## Architecture

### Digital pipeline

```
Screenshot
    → 5-tier board detection (chessboard_grid_segmentation.py)
    → Per-cell perspective warp (each of 64 cells warped independently)
    → Grayscale → background normalise → CLAHE
    → Synthetic CNN classifier  (13 classes: Empty + 6 white + 6 black)
    → FEN
```

### Physical pipeline

```
Photo
    → YOLOv8n-pose corner detection  →  4 corner keypoints
    → Homography (raw image → 8×8 board space)
    → YOLOv8 piece detection on ORIGINAL raw image  →  N bounding boxes
    → Per-piece: crop bbox → CLAHE on HSV V-channel
    → Physical CNN classifier  (12 classes: 6 white + 6 black)
    → Piece centers transformed via homography → square assignment
    → Squares with no detected piece = Empty
    → FEN
```

Both pipelines share the same corner-confirmation UI, board editor, turn selector, and Stockfish analysis panel.

---

## Models

| Model                | Architecture           | Purpose                                    | Val accuracy                   |
| -------------------- | ---------------------- | ------------------------------------------ | ------------------------------ |
| Synthetic classifier | Custom DepthwiseSepCNN | Digital square classification (13 cls)     | 99.00% val / 99.11% test       |
| Physical classifier  | Custom DepthwiseSepCNN | Physical piece classification (12 cls)     | 98.66% val / 97.42% test       |
| Corner detector      | YOLOv8n-pose           | Locate 4 board corners in real photos      | mAP50: 0.995 / mAP50-95: 0.995 |
| Piece detector       | YOLOv8n                | Locate piece bounding boxes (single class) | mAP50: 0.992 / mAP50-95: 0.836 |

Trained classifier weights (`runs/`) and YOLO weights (`models/`) are included in the repository. Full training artifacts (all epoch checkpoints, logs, curves) are on [Google Drive](https://drive.google.com/file/d/1ToSGFEDoMVAqu-FxFT13QL_iTxMkQ_sY/view?usp=sharing).

---

## Project structure

```
Let-It-Chessify/
├── run.py                          # Backend launcher (run from project root)
│
├── backend/
│   ├── README.md                   # Backend-specific docs (pipeline detail, env vars, notes)
│   ├── main.py                     # FastAPI app — all endpoints
│   ├── engine.py                   # Stockfish wrapper
│   ├── requirements.txt
│   ├── __init__.py                 # Marks backend/ as a package (for `backend.*` imports)
│   ├── models/
│   │   └── README.md               # Stockfish binary download + placement instructions
│   └── detection/
│       ├── __init__.py
│       ├── chessboard_grid_segmentation.py  # 5-tier digital board detector
│       ├── classical_cv.py         # Board contour/color/edge detection helpers
│       ├── corner_detector.py      # YOLOv8n-pose wrapper
│       ├── piece_detector.py       # YOLOv8 single-class piece detector
│       ├── board_mapper.py         # Homography + square assignment
│       ├── preprocessing.py        # Two separate preprocessing pipelines
│       ├── classifier.py           # Dual ONNX classifier loader
│       └── fen.py                  # FEN generation + validation
│
├── frontend/
│   ├── README.md                   # Frontend-specific docs (scripts, API contract)
│   ├── src/
│   │   ├── App.jsx                 # 5-step wizard with non-linear navigation
│   │   ├── main.jsx                # React entry point
│   │   ├── index.css
│   │   ├── api/chess.js            # Backend API calls
│   │   ├── hooks/useSquareFit.js   # Responsive board sizing
│   │   └── components/
│   │       ├── UploadStep.jsx      # Image upload + Screenshot/Physical toggle
│   │       ├── CornerConfirmStep.jsx  # Grid overlay + draggable corners
│   │       ├── BoardEditorStep.jsx    # Piece drag/drop editor + palette
│   │       ├── TurnSelectStep.jsx     # Who moves next
│   │       ├── AnalysisStep.jsx       # Score bar + move suggestions
│   │       ├── ScoreBar.jsx           # Vertical centipawn bar
│   │       └── StepIndicator.jsx      # Left sidebar progress
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── models/
│   ├── README.md                   # Which weights are kept vs. archived on Drive, folder layout
│   ├── corner_detection/train-2/weights/best.pt   # YOLOv8n-pose corner model
│   └── piece_detection/train/weights/best.pt      # YOLOv8 piece detector
│
├── runs/
│   ├── runs_README.md              # What's in this folder
│   ├── synthetic/square_classifier.onnx   # Digital CNN (0.08 MB) + best.pt, history.csv, label_map.json
│   └── physical/square_classifier.onnx    # Physical CNN (0.08 MB) + best.pt, history.csv, label_map.json
│
├── train_classifier.py             # Train the square classifier CNNs
├── train_corner_detection.py       # Train the corner detection model
├── train_piece_detection.py        # Train the piece detection model
├── build_splits.py                 # Build train/val/test CSV splits (classifier only)
├── onnx_classify.py                # CLI tool: classify a single square crop
│
├── Synthetic_data_generation scripts/
│   ├── generate_synthetic_data.py  # Board-position renderer (python-chess + cairosvg) → organized_data directly
│   └── download_pieces.py          # Downloads 35 Chess.com piece themes → organized_data directly
│
├── data_organizing_scripts/        # Convert raw_data/ into organized_data/ training format
│   ├── Chess Piece Detection_organize.py     # Kaggle Pascal VOC → physical piece crops
│   ├── FENiT-FEN_organize.py                 # FENiT-FEN → corner labels + physical piece crops
│   ├── chessboard-corner-detect_organize.py  # → corner_detection training layout
│   ├── organize_piece_detection_data.py      # Merges 2 sources → single-class piece-detection YOLO set
│   └── self_synthetic_made_organize.py       # Self-collected digital screenshots → synthetic square crops
│                                              #   (runs them through the real detection pipeline, so
│                                              #    training crops match what /detect sees at inference)
│
├── Training_model_notebooks_from_colab/      # Reference only — the actual Colab notebooks used to train
│   ├── Classification.ipynb                  #   each model on a free T4 GPU. Not a required local step;
│   ├── Corner_Detection.ipynb                #   the .py scripts below do the same job locally/on any GPU.
│   └── Piece_Detection.ipynb
│
├── raw_data/                       # Not in repo (large) — see raw_data/README.md
│   └── README.md
└── organized_data/                 # Not in repo (large) — see organized_data/README.md
    └── README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Stockfish](https://stockfishchess.org/download/) — place `stockfish.exe` (Windows) or `stockfish` (Linux/Mac) in `backend/models/` or anywhere on your `PATH`. Direct download link also in `backend/models/README.md`.

### 1. Clone

```bash
git clone https://github.com/<your-username>/Let-It-Chessify.git
cd Let-It-Chessify
```

### 2. Backend

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Run

Open **two terminals** from the project root:

**Terminal 1 — backend:**

```bash
python run.py
# Backend running at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm run dev
# App running at http://localhost:5173
```

Open `http://localhost:5173` in your browser.

---

## API endpoints

| Endpoint            | Method | Purpose                                            |
| ------------------- | ------ | -------------------------------------------------- |
| `/health`         | GET    | Model load status for all 4 models + Stockfish     |
| `/detect-corners` | POST   | Upload image → detected grid + overlay image      |
| `/refine-corners` | POST   | User-adjusted corners → recomputed grid + overlay |
| `/classify`       | POST   | Confirmed grid → square labels + FEN              |
| `/edit-fen`       | POST   | Board editor changes → updated FEN                |
| `/analyze`        | POST   | Confirmed FEN → Stockfish eval + top moves        |
| `/detect`         | POST   | Legacy one-shot (corners + classify in one call)   |

The frontend uses `/detect-corners` → optional `/refine-corners` → `/classify` for the corner-confirmation flow.

---

## Datasets

Three public datasets, plus one self-collected set, were used for training. The raw data is not in this repository due to size — download from [Google Drive](https://drive.google.com/file/d/1CDZ7xPwHqXZSiGEIdCSVykBbjqqR981P/view?usp=sharing) and place the extracted `raw_data/` folder in the project root.

| Dataset                                                                                                   | Used for                               | Format                                            |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| [Chess Piece Detection](https://www.kaggle.com/datasets/tannergi/chess-piece-detection) (Kaggle)           | Physical piece classifier training     | Pascal VOC XML bboxes                             |
| [chessboard-corner-detect](https://www.kaggle.com/datasets/franciscoana/chessboard-corner-detect) (Kaggle) | Corner detection model training        | YOLO-pose keypoints                               |
| [FENiT-FEN](https://www.kaggle.com/datasets/timotiusdominikus/fenit-fen) (Kaggle)                          | Corner detection + physical classifier | YOLO bboxes (13 classes + board)                  |
| `self_synthetic_made` (self-collected)                                                                  | Digital square classifier training     | Screenshots + a matching FEN label file per image |

`self_synthetic_made` is a self-collected set of screenshots from digital chess interfaces (Lichess, Chess.com, etc.), each paired with a FEN label file. Unlike the fully-rendered synthetic data below, `self_synthetic_made_organize.py` runs these screenshots through the *actual* 5-tier digital board detector (`chessboard_grid_segmentation.py`) to produce training crops — so the classifier trains on the same detection/slicing artifacts it will see at inference, not idealized renders.

Separately, fully-synthetic digital data was generated using `python-chess` + `cairosvg` (clean rendered board positions, no detection pipeline involved) and `download_pieces.py` (35 Chess.com piece themes via CDN). Both scripts write straight into `organized_data/classification/synthetic/` — they don't go through `data_organizing_scripts/`.

---

## Training your own models

Every training script (`train_classifier.py`, `train_corner_detection.py`, `train_piece_detection.py`) reads directly from `organized_data/`. The only real prerequisite is getting that folder populated — once it's there, you just run the scripts.

### Step 1 — Get `organized_data/` in place

Pick whichever option you need:

- **Fastest:** download the pre-built [`organized_data` zip](https://drive.google.com/file/d/1CF8VzC0JuQotPhF-P576zeE_jMEViK-V/view?usp=sharing) and extract it to the project root. Nothing else in this section is required.
- **From the raw datasets:** run the synthetic data generation scripts (`python "Synthetic_data_generation scripts/generate_synthetic_data.py"` and `python "Synthetic_data_generation scripts/download_pieces.py"`, which write rendered/theme crops straight into `organized_data/classification/synthetic/`); then download [`raw_data`](https://drive.google.com/file/d/1CDZ7xPwHqXZSiGEIdCSVykBbjqqR981P/view?usp=sharing), extract to the project root, and run the matching script(s) in `data_organizing_scripts/` for whichever model(s) you're training (e.g. `python "data_organizing_scripts/FENiT-FEN_organize.py"`). Each one writes into the correct `organized_data/...` subfolder.

### Step 2 — Build splits (classifier only)

```bash
python build_splits.py
```

YOLO training doesn't need this — `train_corner_detection.py` / `train_piece_detection.py` read `organized_data/corner_detection/data.yaml` and `organized_data/piece_detection/data.yaml` directly, which the organize scripts already generate.

### Step 3 — Train

```bash
# Classifiers (GPU recommended)
python train_classifier.py --mode synthetic --epochs 40 --batch 64
python train_classifier.py --mode physical  --epochs 40 --batch 64

# YOLO models (GPU strongly recommended)
python train_corner_detection.py
python train_piece_detection.py
```

Each script has its own flags (epochs, batch size, device, resume, etc.) — run with `--help` for the full list.

`Training_model_notebooks_from_colab/` is not a separate step — it's the actual Colab notebooks used to train each model on a free T4 GPU, kept for reference/reproducibility. They run the same `.py` scripts above; there's nothing in them you need that isn't already in this section.

---

## Tech stack

| Layer                      | Technology                          |
| -------------------------- | ----------------------------------- |
| Backend                    | Python · FastAPI · Uvicorn        |
| Board detection (digital)  | OpenCV · 5-tier cascade            |
| Board detection (physical) | Ultralytics YOLOv8n-pose            |
| Piece detection            | Ultralytics YOLOv8n                 |
| Piece classification       | Custom CNN · ONNX Runtime          |
| Chess logic                | python-chess                        |
| Engine                     | Stockfish (local binary)            |
| Frontend                   | React 19 · Vite · Tailwind CSS v4 |
| Board UI                   | react-chessboard v5 · chess.js     |
| Icons                      | Lucide React                        |

---

## Documentation

This README covers the project as a whole. few folders have their own README with more detail:

| README                                                  | Covers                                                                       |
| ------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [`backend/README.md`](backend/README.md)               | Full pipeline breakdown, API endpoints, env var overrides, contributor notes |
| [`backend/models/README.md`](backend/models/README.md) | Stockfish binary download + placement                                        |
| [`frontend/README.md`](frontend/README.md)             | Frontend scripts, API contract, component layout                             |
| [`models/README.md`](models/README.md)                 | Which`.pt` weights are kept in-repo vs. archived, full Drive layout        |
| [`raw_data/README.md`](raw_data/README.md)             | Raw dataset folder structure + download link                                 |
| [`organized_data/README.md`](organized_data/README.md) | Organized/training-ready data structure + download link                      |
| [`runs/runs_README.md`](runs/runs_README.md)           | Classifier run outputs kept in-repo                                          |

---

## Acknowledgements

- [python-chess](https://python-chess.readthedocs.io/) — chess logic and FEN generation
- [Ultralytics](https://ultralytics.com/) — YOLOv8 training and inference
- [react-chessboard](https://github.com/Clariity/react-chessboard) — interactive board component
- [Stockfish](https://stockfishchess.org/) — open-source chess engine
