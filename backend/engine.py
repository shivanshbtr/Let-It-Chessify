"""
Chess OCR — Stockfish Engine Wrapper
======================================
Provides eval bar (centipawn score) and top move suggestions.
Fully offline — uses local Stockfish binary.

Install Stockfish:
    Linux/Mac:  sudo apt install stockfish  OR  brew install stockfish
    Windows:    download from https://stockfishchess.org/download/
                place stockfish.exe in backend/models/ or add to PATH
"""

import chess
import chess.engine
from pathlib import Path


# ── Stockfish binary search paths ─────────────────────────────────────────────
# Searched in order — first found is used

STOCKFISH_PATHS = [
    "stockfish",                          # if in PATH
    "/usr/games/stockfish",               # Linux apt install
    "/usr/local/bin/stockfish",           # Mac brew
    "/opt/homebrew/bin/stockfish",        # Mac M1 brew
    "backend/models/stockfish",           # bundled in project
    "backend/models/stockfish.exe",       # Windows bundled
    "models/stockfish",
]


def find_stockfish():
    """Find Stockfish binary from known paths."""
    import os
    import shutil

    # Explicit override (used by the desktop-app launcher so the packaged
    # .exe doesn't depend on the current working directory) -- checked first.
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Check PATH first
    sf = shutil.which("stockfish")
    if sf:
        return sf
    # Check hardcoded paths
    for path in STOCKFISH_PATHS:
        if Path(path).exists():
            return path
    return None


class StockfishEngine:
    """
    Wrapper around python-chess's UCI engine interface.
    Opened once per request and closed cleanly after.
    """

    def __init__(self, stockfish_path=None, depth=25, move_time=None):
        self.path      = stockfish_path or find_stockfish()
        self.depth     = depth
        self.move_time = move_time   # seconds per analysis; None = depth-only, no time cap

        if self.path is None:
            raise FileNotFoundError(
                "Stockfish not found. Install it with:\n"
                "  Linux: sudo apt install stockfish\n"
                "  Mac:   brew install stockfish\n"
                "  Windows: download from https://stockfishchess.org/download/"
            )

    def analyze(self, fen, turn="w", num_moves=3):
        """
        Analyze a position and return eval + top moves.

        Args:
            fen:       FEN string of the position
            turn:      "w" or "b" — whose turn to move
            num_moves: number of top moves to return

        Returns dict:
            {
                "eval_cp":     int,    # centipawn eval from current player's perspective
                "eval_type":   str,    # "cp" or "mate"
                "mate_in":     int or None,
                "best_moves":  [
                    {
                        "uci":       str,   # e.g. "e2e4"
                        "san":       str,   # e.g. "e4"
                        "eval_cp":   int,
                        "eval_type": str,
                        "mate_in":   int or None,  # moves to mate, if eval_type == "mate"
                    }, ...
                ],
                "score_bar":   float,  # -1.0 to 1.0 for UI score bar
            }
        """
        try:
            board = chess.Board(fen)
        except Exception as e:
            raise ValueError(f"Invalid FEN: {e}")

        limit = chess.engine.Limit(depth=self.depth, time=self.move_time) if self.move_time else chess.engine.Limit(depth=self.depth)

        with chess.engine.SimpleEngine.popen_uci(self.path) as engine:
            # Analyze with multipv for top N moves
            info = engine.analyse(
                board,
                limit,
                multipv=num_moves,
            )

        if not info:
            return _empty_result()

        # Primary eval (best move)
        primary = info[0]
        score   = primary.get("score")

        if score is None:
            return _empty_result()

        # Eval from white's perspective for score bar
        white_pov = score.white()

        eval_cp   = None
        eval_type = "cp"
        mate_in   = None

        if white_pov.is_mate():
            eval_type = "mate"
            mate_in   = white_pov.mate()
            if mate_in == 0:
                # Already checkmate (0 has no sign in Python, so mate_in > 0
                # is always False here) -- the side to move is the one
                # that's mated, so infer the winner from board.turn instead.
                eval_cp = -10000 if board.turn == chess.WHITE else 10000
            else:
                eval_cp = 10000 if mate_in > 0 else -10000
        else:
            eval_cp = white_pov.score()

        # Score bar: sigmoid-like mapping of centipawns → [-1, 1]
        score_bar = cp_to_bar(eval_cp)

        # Top moves
        best_moves = []
        for pv_info in info[:num_moves]:
            pv    = pv_info.get("pv", [])
            if not pv:
                continue
            move  = pv[0]
            mv_score = pv_info.get("score")

            mv_cp   = None
            mv_type = "cp"
            mv_mate = None
            if mv_score:
                mv_white = mv_score.white()
                if mv_white.is_mate():
                    mv_type = "mate"
                    mv_mate = mv_white.mate()
                    if mv_mate == 0:
                        # Same signless-zero issue as above: this move itself
                        # delivers mate. Infer the winner from whose turn it
                        # is right after the move (that side is the mated one).
                        board_after = board.copy()
                        board_after.push(move)
                        mv_cp = -10000 if board_after.turn == chess.WHITE else 10000
                    else:
                        mv_cp = 10000 if mv_mate > 0 else -10000
                else:
                    mv_cp = mv_white.score()

            try:
                san = board.san(move)
            except Exception:
                san = move.uci()

            best_moves.append({
                "uci":       move.uci(),
                "san":       san,
                "eval_cp":   mv_cp,
                "eval_type": mv_type,
                "mate_in":   abs(mv_mate) if mv_mate is not None else None,
            })

        return {
            "eval_cp":    eval_cp,
            "eval_type":  eval_type,
            "mate_in":    mate_in,
            "best_moves": best_moves,
            "score_bar":  score_bar,
        }

    def analyze_stream(self, fen, turn="w", num_moves=3):
        """
        Generator version of analyze(): yields a result dict after every
        depth update Stockfish reports during iterative deepening, with an
        extra "depth" key (and "done": True on the final yield). Lets a
        caller show live progress -- depth counting up, eval/best-moves
        refining in real time -- instead of waiting for one final result.

        Each yielded dict has the same shape as analyze()'s return value,
        plus "depth" and "done".
        """
        try:
            board = chess.Board(fen)
        except Exception as e:
            raise ValueError(f"Invalid FEN: {e}")

        limit = (
            chess.engine.Limit(depth=self.depth, time=self.move_time)
            if self.move_time else
            chess.engine.Limit(depth=self.depth)
        )

        # Lines for the depth currently being swept (multipv index -> info).
        # Stockfish reports line 1, then line 2, then line 3, etc. for one
        # depth before moving to the next depth's line 1. We only emit once
        # a full sweep for a depth is genuinely complete -- signaled by line
        # 1 of the *next* depth arriving -- so every snapshot has moves that
        # all belong to the same depth. Emitting as soon as line 1 updates
        # (the old approach) mixed a fresh #1 move with stale #2/#3 moves
        # still left over from the previous depth, which showed up as
        # arrows jumping to the wrong squares / flickering between
        # unrelated moves.
        pending_lines = {}
        pending_depth = None

        with chess.engine.SimpleEngine.popen_uci(self.path) as engine:
            with engine.analysis(board, limit, multipv=num_moves) as analysis:
                for info in analysis:
                    depth = info.get("depth")
                    if depth is None or "score" not in info:
                        continue
                    pv_index = info.get("multipv", 1)

                    if pv_index == 1 and pending_lines:
                        # Line 1 of a new depth arriving means the previous
                        # depth's sweep just finished -- emit it now, then
                        # start accumulating the new depth fresh.
                        snapshot = self._snapshot_from_lines(board, pending_lines)
                        snapshot["depth"] = pending_depth
                        snapshot["done"] = False
                        yield snapshot
                        pending_lines = {}

                    pending_lines[pv_index] = info
                    pending_depth = depth

        if pending_lines:
            final = self._snapshot_from_lines(board, pending_lines)
            final["depth"] = pending_depth
            final["done"] = True
            yield final

    def _snapshot_from_lines(self, board, lines):
        """Build one analyze()-shaped result dict from the multipv lines
        accumulated so far during streaming analysis."""
        primary = lines.get(1)
        if primary is None:
            return _empty_result()

        score = primary.get("score")
        if score is None:
            return _empty_result()

        white_pov = score.white()
        eval_type = "cp"
        mate_in   = None

        if white_pov.is_mate():
            eval_type = "mate"
            mate_in   = white_pov.mate()
            if mate_in == 0:
                eval_cp = -10000 if board.turn == chess.WHITE else 10000
            else:
                eval_cp = 10000 if mate_in > 0 else -10000
        else:
            eval_cp = white_pov.score()

        score_bar = cp_to_bar(eval_cp)

        best_moves = []
        for idx in sorted(lines.keys()):
            pv_info = lines[idx]
            pv = pv_info.get("pv", [])
            if not pv:
                continue
            move = pv[0]
            mv_score = pv_info.get("score")

            mv_cp   = None
            mv_type = "cp"
            mv_mate = None
            if mv_score:
                mv_white = mv_score.white()
                if mv_white.is_mate():
                    mv_type = "mate"
                    mv_mate = mv_white.mate()
                    if mv_mate == 0:
                        board_after = board.copy()
                        board_after.push(move)
                        mv_cp = -10000 if board_after.turn == chess.WHITE else 10000
                    else:
                        mv_cp = 10000 if mv_mate > 0 else -10000
                else:
                    mv_cp = mv_white.score()

            try:
                san = board.san(move)
            except Exception:
                san = move.uci()

            best_moves.append({
                "uci":       move.uci(),
                "san":       san,
                "eval_cp":   mv_cp,
                "eval_type": mv_type,
                "mate_in":   abs(mv_mate) if mv_mate is not None else None,
            })

        return {
            "eval_cp":    eval_cp,
            "eval_type":  eval_type,
            "mate_in":    mate_in,
            "best_moves": best_moves,
            "score_bar":  score_bar,
        }


def cp_to_bar(cp):
    """
    Map centipawn score to score bar value in [-1.0, 1.0].
    Uses sigmoid-like mapping:
      0 cp   → 0.0  (equal)
      300 cp → ~0.6 (slight advantage)
      1000cp → ~0.9 (winning)
    White winning → positive, black winning → negative.
    """
    if cp is None:
        return 0.0
    import math
    # Clamp to ±10000 (mate scores)
    cp = max(-10000, min(10000, cp))
    # Sigmoid: 2 / (1 + exp(-cp/400)) - 1
    return 2.0 / (1.0 + math.exp(-cp / 400.0)) - 1.0


def _empty_result():
    return {
        "eval_cp":    0,
        "eval_type":  "cp",
        "mate_in":    None,
        "best_moves": [],
        "score_bar":  0.0,
    }
