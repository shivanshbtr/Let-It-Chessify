# Let It Chessify — Frontend

A React + Vite app that turns a photo (or screenshot) of a chess board into an analysable position: it detects the board corners, classifies each square's piece, lets you fix any mistakes, and then hands the position off to an engine for evaluation, move suggestions, and full move-by-move play. You can also skip the photo entirely and start from the standard opening position, or import an existing game from a PGN.

## Tech stack

- React 19 + Vite
- Tailwind CSS v4
- [chess.js](https://github.com/jhlywa/chess.js) for move legality, PGN parsing/generation, and FEN handling
- [react-chessboard](https://github.com/Clariity/react-chessboard) for the interactive board
- lucide-react for icons

## Prerequisites

- Node.js 20+ and npm
- A running backend API (see [Backend](#backend) below) — this is the frontend only

## Getting started

```bash
# 1. Install dependencies — required before the first run.
#    node_modules/ is not checked into this repo .
#    so `npm run dev` will fail with "vite: not found" until this
#    has been run at least once, and again whenever dependencies change.
npm install

# 2. Start the dev server
npm run dev
```

The dev server runs on Vite's default port (`http://localhost:5173`). Any request to `/api/*` is proxied to `http://localhost:8000` (see `vite.config.js`), so make sure the backend is running there first — otherwise uploads, classification, and analysis calls will fail.

### Other scripts

```bash
npm run build     # production build, output to dist/
npm run preview   # preview the production build locally
npm run lint      # oxlint
```

## Backend

The frontend expects a backend exposing the following endpoints under `/api` (proxied to `http://localhost:8000` in dev — see `src/api/chess.js`):

| Endpoint                 | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `POST /detect-corners` | Detect the board and its four corners from an uploaded image |
| `POST /refine-corners` | Re-run detection with manually adjusted corners              |
| `POST /classify`       | Classify each square's piece from the confirmed grid         |
| `POST /edit-fen`       | Turn edited square labels + turn into a FEN                  |
| `POST /analyze`        | Evaluate a position and return best-move suggestions         |

This backend is not part of this repository.

## Project structure

```
src/
  api/chess.js              # backend API client
  components/
    UploadStep.jsx          # photo/screenshot upload, start-from-scratch, PGN import
    CornerConfirmStep.jsx   # manual corner adjustment
    BoardEditorStep.jsx     # per-square piece correction
    TurnSelectStep.jsx      # whose move it is
    AnalysisStep.jsx        # board, eval, suggestions, move history, PGN export
    ScoreBar.jsx
    StepIndicator.jsx       # left-hand step navigation
  hooks/useSquareFit.js
  App.jsx                   # step/state orchestration
  main.jsx
```

## Notes

- `node_modules/` and `dist/` are excluded— always run `npm install` after cloning or pulling dependency changes.
