"""
Chess OCR — FEN Generation
=============================
Works on a unified square_labels dict {square_index -> label_string},
where label_string is one of the 13 synthetic-style labels:
    Empty, wK, wQ, wR, wB, wN, wP, bK, bQ, bR, bB, bN, bP

For the physical pipeline, "Empty" is injected via the confidence
threshold step (confidence_threshold.py) BEFORE this module is called,
so fen.py never needs to know which pipeline produced the labels.
"""

import chess


LABEL_TO_PIECE = {
    "wK": chess.Piece(chess.KING,   chess.WHITE),
    "wQ": chess.Piece(chess.QUEEN,  chess.WHITE),
    "wR": chess.Piece(chess.ROOK,   chess.WHITE),
    "wB": chess.Piece(chess.BISHOP, chess.WHITE),
    "wN": chess.Piece(chess.KNIGHT, chess.WHITE),
    "wP": chess.Piece(chess.PAWN,   chess.WHITE),
    "bK": chess.Piece(chess.KING,   chess.BLACK),
    "bQ": chess.Piece(chess.QUEEN,  chess.BLACK),
    "bR": chess.Piece(chess.ROOK,   chess.BLACK),
    "bB": chess.Piece(chess.BISHOP, chess.BLACK),
    "bN": chess.Piece(chess.KNIGHT, chess.BLACK),
    "bP": chess.Piece(chess.PAWN,   chess.BLACK),
    "Empty": None,
}

PIECE_TO_LABEL = {v: k for k, v in LABEL_TO_PIECE.items() if v is not None}


def labels_to_fen(square_labels, turn="w"):
    """
    Args:
        square_labels: {square_index -> label_string}
        turn: "w" or "b"

    Returns: (fen_string, warnings_list)
    """
    board = chess.Board(fen=None)

    for sq_idx, label in square_labels.items():
        piece = LABEL_TO_PIECE.get(label)
        if piece is not None:
            board.set_piece_at(sq_idx, piece)

    board.turn = chess.WHITE if turn == "w" else chess.BLACK

    warnings = validate_position(board)
    return board.fen(), warnings


def validate_position(board):
    warnings = []

    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))

    if white_kings == 0:
        warnings.append("No white king detected — add one in the editor")
    if white_kings > 1:
        warnings.append(f"Multiple white kings detected ({white_kings}) — check editor")
    if black_kings == 0:
        warnings.append("No black king detected — add one in the editor")
    if black_kings > 1:
        warnings.append(f"Multiple black kings detected ({black_kings}) — check editor")

    for sq in chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)):
        if chess.square_rank(sq) == 7:
            warnings.append("White pawn on rank 8 — likely misclassified piece")
            break
    for sq in chess.SquareSet(board.pieces(chess.PAWN, chess.BLACK)):
        if chess.square_rank(sq) == 0:
            warnings.append("Black pawn on rank 1 — likely misclassified piece")
            break

    return warnings


def fen_to_square_labels(fen):
    """Parse a FEN string back into a square_labels dict (for editor round-trip)."""
    board  = chess.Board(fen)
    labels = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        labels[sq] = "Empty" if piece is None else PIECE_TO_LABEL.get(piece, "Empty")
    return labels


def get_turn_from_fen(fen):
    parts = fen.split()
    return parts[1] if len(parts) > 1 else "w"
