import { useState, useMemo } from 'react'
import { ChevronRight, ArrowLeft } from 'lucide-react'

// Piece sitting on a given square (e.g. 'e1') within a FEN board-part
// string. Deliberately doesn't use chess.js here -- that validates the
// whole position strictly (exactly one king per side, etc.) and would
// throw on positions the backend only warns about rather than blocks.
// This just needs to look at specific squares, nothing more.
function pieceAt(boardFen, square) {
  const file = square.charCodeAt(0) - 97           // a-h -> 0-7
  const rankFromTop = 8 - parseInt(square[1], 10)  // rank 8 -> row 0 ... rank 1 -> row 7
  const row = boardFen.split('/')[rankFromTop]
  if (!row) return null
  let col = 0
  for (const ch of row) {
    if (/\d/.test(ch)) {
      col += parseInt(ch, 10)
    } else {
      if (col === file) return ch
      col += 1
    }
  }
  return null
}

const CASTLING_ORDER = ['K', 'Q', 'k', 'q']

const PIECE_GLYPHS = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
}

// Full 8x8 grid (row 0 = rank 8, row 7 = rank 1) for the mini-board
// display -- pieceAt() above only checks single squares on demand, this
// expands the whole board-part string at once for rendering.
function parseBoardGrid(boardFen) {
  return boardFen.split('/').map((row) => {
    const cells = []
    for (const ch of row) {
      if (/\d/.test(ch)) {
        for (let i = 0; i < parseInt(ch, 10); i++) cells.push(null)
      } else {
        cells.push(ch)
      }
    }
    return cells
  })
}

// Geometrically-possible en passant target squares given whose turn it is.
// Only ever ONE en passant target can exist in a real FEN at a time (this
// is a single optional value, not independent flags like castling), but
// a position can easily have multiple simultaneously-plausible candidates
// (e.g. two enemy pawns adjacent to two different just-pushed pawns) --
// hence returning a list for a single-select UI, not a boolean per square.
// Restricted to candidates where an adjacent enemy pawn could actually
// use it; a "double-pushed pawn with nothing beside it" isn't worth
// offering since selecting it would be inert either way.
function findEnPassantCandidates(boardFen, turn) {
  const results = []
  const rank = turn === 'w' ? '5' : '4'
  const targetRank = turn === 'w' ? '6' : '3'
  const originRank = turn === 'w' ? '7' : '2'   // the pawn's own home rank
  const pawnChar = turn === 'w' ? 'p' : 'P'   // the side that just double-pushed
  const adjChar = turn === 'w' ? 'P' : 'p'    // the side that would capture
  for (let file = 0; file < 8; file++) {
    const fileChar = String.fromCharCode(97 + file)
    const sq = fileChar + rank
    if (pieceAt(boardFen, sq) !== pawnChar) continue

    // A pawn can only have JUST double-pushed if its origin square (two
    // ranks further back, on its own home rank) is empty. If something's
    // sitting there, this pawn can't have moved from there on the
    // immediately preceding move -- it's simply been on this rank for a
    // while, and en passant no longer applies.
    if (pieceAt(boardFen, fileChar + originRank) !== null) continue

    const targetSq = fileChar + targetRank
    // The square it passed over (the en passant target itself) must also
    // have been empty for the double push to have been legal at all.
    if (pieceAt(boardFen, targetSq) !== null) continue

    const left = file - 1, right = file + 1
    const hasAdjacentCapturer =
      (left >= 0 && pieceAt(boardFen, String.fromCharCode(97 + left) + rank) === adjChar) ||
      (right <= 7 && pieceAt(boardFen, String.fromCharCode(97 + right) + rank) === adjChar)
    if (hasAdjacentCapturer) {
      results.push({ square: targetSq, pawnSquare: sq })
    }
  }
  return results
}

export default function TurnSelectStep({ fen, onConfirm, onBack }) {
  // Initialize from the FEN's own side-to-move field, so coming back to this
  // step (e.g. from Analysis) shows whatever was previously selected instead
  // of always resetting to White.
  const [turn, setTurn] = useState(() => (fen.split(' ')[1] === 'b' ? 'b' : 'w'))

  // Which of the 4 castling rights are even geometrically possible, based
  // on whether the king and the relevant rook are actually sitting on
  // their home squares in the OCR'd position. A static image can't know
  // true game history (a rook could've moved away and back), so this is
  // "possible, not proven" -- exactly the same limitation any chess-board
  // editor has. Only possible rights get offered as checkboxes at all;
  // there's nothing to confirm for a right that's already impossible.
  const possible = useMemo(() => {
    const boardFen = fen.split(' ')[0]
    return {
      K: pieceAt(boardFen, 'e1') === 'K' && pieceAt(boardFen, 'h1') === 'R',
      Q: pieceAt(boardFen, 'e1') === 'K' && pieceAt(boardFen, 'a1') === 'R',
      k: pieceAt(boardFen, 'e8') === 'k' && pieceAt(boardFen, 'h8') === 'r',
      q: pieceAt(boardFen, 'e8') === 'k' && pieceAt(boardFen, 'a8') === 'r',
    }
  }, [fen])

  const anyPossible = CASTLING_ORDER.some((k) => possible[k])

  // Defaults to checked wherever geometrically possible -- the common
  // case (early/midgame, nothing's moved yet) -- user unchecks whichever
  // they know has actually already been used.
  const [castling, setCastling] = useState(() => ({ ...possible }))

  const toggleCastling = (key) => {
    setCastling((c) => ({ ...c, [key]: !c[key] }))
  }

  // Recomputed on turn change (unlike castling's `possible`) -- who could
  // have just double-pushed depends entirely on whose move it is next.
  const epCandidates = useMemo(
    () => findEnPassantCandidates(fen.split(' ')[0], turn),
    [fen, turn]
  )
  const epCandidateSquares = useMemo(
    () => new Set(epCandidates.map((c) => c.square)),
    [epCandidates]
  )
  const boardGrid = useMemo(() => parseBoardGrid(fen.split(' ')[0]), [fen])

  // Single-select, defaults to "None" even when candidates exist -- a
  // screenshot can't actually know whether a pawn just double-pushed or
  // has been sitting there for several moves, so silence is right far
  // more often than guessing. Selecting a turn resets this, since
  // whatever was chosen may not even be a valid candidate for the new
  // side to move.
  const [epSquare, setEpSquare] = useState(null)

  const selectTurn = (t) => {
    setTurn(t)
    setEpSquare(null)
  }

  // Update FEN with selected turn, castling rights, AND en passant.
  // Previously this only ever touched the turn field, leaving castling
  // permanently at "-" (disabled) for every position, even a fresh
  // starting position with king and both rooks untouched -- Stockfish
  // was silently being told castling was illegal in cases where it
  // obviously wasn't.
  const getFenWithTurn = (t, c = castling, ep = epSquare) => {
    const parts = fen.split(' ')
    parts[1] = t
    parts[2] = CASTLING_ORDER.filter((k) => possible[k] && c[k]).join('') || '-'
    parts[3] = ep || '-'
    return parts.join(' ')
  }

  const turnSection = (
    <section className="flex flex-col items-center gap-3">
      <h2 className="font-display text-lg font-semibold text-[#F5F0E8]">
        Who moves next?
      </h2>
      <div className="flex gap-4">
        <button
          onClick={() => selectTurn('w')}
          className={`flex flex-col items-center gap-2 px-6 py-5 rounded-2xl border-2
                      transition-all duration-200 ${
            turn === 'w'
              ? 'border-[#F5F0E8] bg-[#F5F0E8]/10 scale-105'
              : 'border-[#333] bg-[#242424] hover:border-[#555]'
          }`}
        >
          <span className="text-4xl">♚</span>
          <span
            className="text-sm font-medium px-3 py-0.5 rounded-full"
            style={{
              background: turn === 'w' ? '#F5F0E8' : '#8A8A8A',
              color: '#1a1a1a',
            }}
          >
            White
          </span>
        </button>

        <button
          onClick={() => selectTurn('b')}
          className={`flex flex-col items-center gap-2 px-6 py-5 rounded-2xl border-2
                      transition-all duration-200 ${
            turn === 'b'
              ? 'border-[#F5F0E8] bg-[#F5F0E8]/10 scale-105'
              : 'border-[#333] bg-[#242424] hover:border-[#555]'
          }`}
        >
          <span className="text-4xl">♔</span>
          <span
            className="text-sm font-medium px-3 py-0.5 rounded-full"
            style={{
              background: '#1a1a1a',
              color: turn === 'b' ? '#F5F0E8' : '#8A8A8A',
            }}
          >
            Black
          </span>
        </button>
      </div>
    </section>
  )

  const castlingSection = (
    <section className="flex flex-col items-center gap-3">
      <h2 className="font-display text-lg font-semibold text-[#F5F0E8]">
        Castling still available?
      </h2>
      <div className="flex flex-col gap-3">
        {(possible.K || possible.Q) && (
          <div className="flex flex-col gap-1.5 px-5 py-3 rounded-2xl border-2 border-[#333] bg-[#242424]">
            <span className="text-[#F5F0E8] text-xs font-medium uppercase tracking-wider">White</span>
            {possible.K && (
              <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                <input
                  type="checkbox"
                  checked={castling.K}
                  onChange={() => toggleCastling('K')}
                  className="accent-[#D8D4C8]"
                />
                Kingside (O-O)
              </label>
            )}
            {possible.Q && (
              <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                <input
                  type="checkbox"
                  checked={castling.Q}
                  onChange={() => toggleCastling('Q')}
                  className="accent-[#D8D4C8]"
                />
                Queenside (O-O-O)
              </label>
            )}
          </div>
        )}
        {(possible.k || possible.q) && (
          <div className="flex flex-col gap-1.5 px-5 py-3 rounded-2xl border-2 border-[#333] bg-[#242424]">
            <span className="text-[#F5F0E8] text-xs font-medium uppercase tracking-wider">Black</span>
            {possible.k && (
              <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                <input
                  type="checkbox"
                  checked={castling.k}
                  onChange={() => toggleCastling('k')}
                  className="accent-[#D8D4C8]"
                />
                Kingside (O-O)
              </label>
            )}
            {possible.q && (
              <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                <input
                  type="checkbox"
                  checked={castling.q}
                  onChange={() => toggleCastling('q')}
                  className="accent-[#D8D4C8]"
                />
                Queenside (O-O-O)
              </label>
            )}
          </div>
        )}
      </div>
    </section>
  )

  return (
    <div className="flex flex-col items-center justify-center min-h-full gap-6 px-4 py-10 relative">
      {onBack && (
        <button
          onClick={onBack}
          className="absolute top-6 left-6 flex items-center gap-1.5 text-[#8A8A8A]
                     hover:text-[#F5F0E8] text-sm transition-colors"
        >
          <ArrowLeft size={14} />
          Back
        </button>
      )}

      <div className="text-center">
        <h1 className="font-display text-3xl font-semibold text-[#F5F0E8] mb-2">
          Set up the position
        </h1>
        <p className="text-[#8A8A8A] text-sm">
          Confirm the side to move and any remaining castling rights.
        </p>
      </div>

      {/* Turn + castling side by side when castling applies (saves
          vertical space, avoids scrolling) -- turn alone, centered,
          when it doesn't. */}
      {anyPossible ? (
        <div className="w-full max-w-2xl flex items-center justify-center gap-6">
          <div className="flex-1 flex justify-center">{turnSection}</div>
          <div className="w-px self-stretch bg-[#333]" />
          <div className="flex-1 flex justify-center">{castlingSection}</div>
        </div>
      ) : (
        turnSection
      )}

      {/* En passant -- a mini read-only board with candidate squares
          highlighted, click to select. Still fundamentally a
          single-select (a FEN can only hold one en passant target even
          when several squares are geometrically plausible, e.g. two
          enemy pawns each adjacent to a different just-pushed pawn) --
          clicking the already-selected square deselects back to None,
          which is the default even when candidates exist. A static
          image can't actually know whether a pawn just double-pushed or
          has simply been sitting there, so silence is the safer default. */}
      {epCandidates.length > 0 && (
        <>
          <div className="w-full max-w-lg h-px bg-[#333]" />
          <section className="w-full max-w-lg flex flex-col items-center gap-2">
            <h2 className="font-display text-lg font-semibold text-[#F5F0E8]">
              En passant available?
            </h2>
            <p className="text-[#8A8A8A] text-xs text-center max-w-xs">
              {epSquare
                ? `${epSquare} is eligible for en passant. Tap again to clear.`
                : "If any of the shown moves was the opponent's just-previous move, tap the highlighted square behind it."}
            </p>

            <div
              className="relative w-48 h-48 rounded-lg overflow-hidden"
              style={{ outline: '2px solid #333', outlineOffset: '-2px' }}
            >
              {/* Layer 1: board squares + pieces (bottom). Left at their
                  natural colors -- the highlight lives entirely in the
                  wash layer above, so it's fine (expected, even) that a
                  translucent wash reads a little differently over the
                  light vs. dark square underneath, the same way it does
                  on any chess site. */}
              <div
                className="grid w-full h-full"
                style={{ gridTemplateColumns: 'repeat(8, 1fr)', gridTemplateRows: 'repeat(8, 1fr)' }}
              >
                {boardGrid.map((row, rowIdx) =>
                  row.map((piece, colIdx) => {
                    const square = String.fromCharCode(97 + colIdx) + (8 - rowIdx)
                    const isLight = (rowIdx + colIdx) % 2 === 0
                    const isCandidate = epCandidateSquares.has(square)
                    const isSelected = epSquare === square
                    return (
                      <button
                        key={square}
                        type="button"
                        disabled={!isCandidate}
                        onClick={() => setEpSquare(isSelected ? null : square)}
                        className="relative flex items-center justify-center text-base leading-none"
                        style={{
                          backgroundColor: isLight ? '#a58e69' : '#7e6351',
                          cursor: isCandidate ? 'pointer' : 'default',
                        }}
                      >
                        {piece && (
                          <span style={{ color: piece === piece.toUpperCase() ? '#f4f3f2' : '#1a1a1a',
                            fontSize: '20px'
                          }}>
                            {PIECE_GLYPHS[piece]}
                          </span>
                        )}
                      </button>
                    )
                  })
                )}
              </div>

              {/* Layer 2: a single accent color (the app's own brass/gold)
                  used as a translucent wash over the whole square --
                  dim for an unselected candidate, a stronger fill with a
                  bright edge once selected so the before/after click is
                  clearly visible even against a light square whose base
                  tone is already close to brass. Clipped with the same
                  rounding as the board so it can't poke past the corner
                  border on edge squares. */}
              <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
                {boardGrid.map((row, rowIdx) =>
                  row.map((piece, colIdx) => {
                    const square = String.fromCharCode(97 + colIdx) + (8 - rowIdx)
                    const isLight = (rowIdx + colIdx) % 2 === 0
                    const isCandidate = epCandidateSquares.has(square)
                    const isSelected = epSquare === square
                    if (!isCandidate) return null
                    const cell = 24
                    const opacity = isSelected ? 0.97 : (isLight ? 0.55 : 0.5)
                    return (
                      <span
                        key={square}
                        className="absolute"
                        style={{
                          left: colIdx * cell,
                          top: rowIdx * cell,
                          width: cell,
                          height: cell,
                          backgroundColor: `rgba(201,161,92,${opacity})`,
                          boxShadow: isSelected ? 'inset 0 0 0 2px rgba(245,240,232,0.9)' : 'none',
                        }}
                      />
                    )
                  })
                )}
              </div>

              {/* Layer 3: arrows from each just-pushed candidate pawn's home
                  rank to its current square, showing the double push that
                  was just played. Drawn as a solid filled shape (shaft +
                  head) rather than a stroked line, in a deeper brass tone
                  so it reads clearly against the lighter gold wash below
                  it. Hidden once a square is selected. */}
              {!epSquare && (
                <svg
                  className="absolute inset-0 pointer-events-none"
                  viewBox="0 0 192 192"
                  width="192"
                  height="192"
                >
                  {epCandidates.map(({ square, pawnSquare }) => {
                    const cell = 24
                    const centerOf = (sq) => {
                      const file = sq.charCodeAt(0) - 97
                      const rank = parseInt(sq[1], 10)
                      const rowIdx = 8 - rank
                      return { x: file * cell + cell / 2, y: rowIdx * cell + cell / 2 }
                    }
                    // The pawn's own home rank -- two ranks behind its
                    // current square -- is where it just double-pushed
                    // from, so the arrow spans that whole move rather
                    // than the single square to the ep target.
                    const originRank = turn === 'w' ? '7' : '2'
                    const originSquare = pawnSquare[0] + originRank
                    const from = centerOf(originSquare)
                    const to = centerOf(pawnSquare)
                    // Start exactly at the origin square's center; shorten
                    // only the destination end so the tip sits just shy of
                    // the pawn's square center rather than landing right
                    // on top of the piece.
                    const dx = to.x - from.x
                    const dy = to.y - from.y
                    const len = Math.hypot(dx, dy) || 1
                    const endPad = 9
                    const tipX = to.x - (dx / len) * endPad
                    const tipY = to.y - (dy / len) * endPad

                    // Build a tapered arrow polygon (shaft + triangular
                    // head) instead of a stroked line + marker, so it
                    // reads as one solid shape.
                    const ux = dx / len
                    const uy = dy / len
                    const px = -uy
                    const py = ux
                    const shaftHalf = 1.6
                    const headHalf = 4.5
                    const headLength = 8
                    const headBaseX = tipX - ux * headLength
                    const headBaseY = tipY - uy * headLength
                    const points = [
                      [from.x + px * shaftHalf, from.y + py * shaftHalf],
                      [headBaseX + px * shaftHalf, headBaseY + py * shaftHalf],
                      [headBaseX + px * headHalf, headBaseY + py * headHalf],
                      [tipX, tipY],
                      [headBaseX - px * headHalf, headBaseY - py * headHalf],
                      [headBaseX - px * shaftHalf, headBaseY - py * shaftHalf],
                      [from.x - px * shaftHalf, from.y - py * shaftHalf],
                    ].map((p) => p.join(',')).join(' ')

                    return (
                      <polygon
                        key={`${pawnSquare}-${square}`}
                        points={points}
                        fill="#7C6640"
                        stroke="#F5F0E8"
                        strokeWidth="0.5"
                        strokeOpacity="0.4"
                      />
                    )
                  })}
                </svg>
              )}
            </div>
          </section>
        </>
      )}

      {/* FEN display */}
      <div className="w-full max-w-lg bg-[#242424] rounded-xl border border-[#333] p-3">
        <p className="text-[#8A8A8A] text-xs uppercase tracking-wider mb-1.5">FEN</p>
        <p className="text-[#F5F0E8] text-xs font-mono break-all leading-relaxed">
          {getFenWithTurn(turn)}
        </p>
      </div>

      <button
        onClick={() => onConfirm(turn, getFenWithTurn(turn))}
        className="flex items-center gap-2 px-8 py-3 rounded-xl bg-[#6B9E6B]
                   hover:bg-[#7aaf7a] text-white font-medium transition-all"
      >
        Analyse position
        <ChevronRight size={18} />
      </button>
    </div>
  )
}
