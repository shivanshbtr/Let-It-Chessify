import { useState, useMemo } from 'react'
import { ChevronRight, ArrowLeft } from 'lucide-react'

// Piece sitting on a given square (e.g. 'e1') within a FEN board-part
// string. Deliberately doesn't use chess.js here -- that validates the
// whole position strictly (exactly one king per side, etc.) and would
// throw on positions the backend only warns about rather than blocks.
// This just needs to look at 6 specific squares, nothing more.
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

  // Update FEN with selected turn AND castling rights. Previously this
  // only ever touched the turn field, leaving castling permanently at
  // "-" (disabled) for every position, even a fresh starting position
  // with king and both rooks untouched -- Stockfish was silently being
  // told castling was illegal in cases where it obviously wasn't.
  const getFenWithTurn = (t, c = castling) => {
    const parts = fen.split(' ')
    parts[1] = t
    parts[2] = CASTLING_ORDER.filter((k) => possible[k] && c[k]).join('') || '-'
    return parts.join(' ')
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-full gap-10 px-4 py-12 relative">
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

      {/* Section 1: Turn selector */}
      <section className="w-full max-w-lg flex flex-col items-center gap-4">
        <h2 className="font-display text-lg font-semibold text-[#F5F0E8]">
          Who moves next?
        </h2>
        <div className="flex gap-4">
          <button
            onClick={() => setTurn('w')}
            className={`flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border-2
                        transition-all duration-200 ${
              turn === 'w'
                ? 'border-[#F5F0E8] bg-[#F5F0E8]/10 scale-105'
                : 'border-[#333] bg-[#242424] hover:border-[#555]'
            }`}
          >
            <span className="text-5xl">♔</span>
            <span className={`text-sm font-medium ${turn === 'w' ? 'text-[#F5F0E8]' : 'text-[#8A8A8A]'}`}>
              White
            </span>
          </button>

          <button
            onClick={() => setTurn('b')}
            className={`flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border-2
                        transition-all duration-200 ${
              turn === 'b'
                ? 'border-[#6B9E6B] bg-[#6B9E6B]/10 scale-105'
                : 'border-[#333] bg-[#242424] hover:border-[#555]'
            }`}
          >
            <span className="text-5xl">♚</span>
            <span className={`text-sm font-medium ${turn === 'b' ? 'text-[#F5F0E8]' : 'text-[#8A8A8A]'}`}>
              Black
            </span>
          </button>
        </div>
      </section>

      {/* Divider between the two equally-weighted sections */}
      {anyPossible && <div className="w-full max-w-lg h-px bg-[#333]" />}

      {/* Section 2: Castling rights -- only shown when at least one is
          geometrically possible given where the king/rooks actually are.
          Given the same heading treatment as "Who moves next?" above so
          neither section reads as the primary one. */}
      {anyPossible && (
        <section className="w-full max-w-lg flex flex-col items-center gap-4">
          <h2 className="font-display text-lg font-semibold text-[#F5F0E8]">
            Castling still available?
          </h2>
          <div className="flex justify-center gap-8">
            {(possible.K || possible.Q) && (
              <div className="flex flex-col gap-2 px-6 py-4 rounded-2xl border-2 border-[#333] bg-[#242424]">
                <span className="text-[#F5F0E8] text-xs font-medium mb-0.5 uppercase tracking-wider">White</span>
                {possible.K && (
                  <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                    <input
                      type="checkbox"
                      checked={castling.K}
                      onChange={() => toggleCastling('K')}
                      className="accent-[#6B9E6B]"
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
                      className="accent-[#6B9E6B]"
                    />
                    Queenside (O-O-O)
                  </label>
                )}
              </div>
            )}
            {(possible.k || possible.q) && (
              <div className="flex flex-col gap-2 px-6 py-4 rounded-2xl border-2 border-[#333] bg-[#242424]">
                <span className="text-[#F5F0E8] text-xs font-medium mb-0.5 uppercase tracking-wider">Black</span>
                {possible.k && (
                  <label className="flex items-center gap-2 text-sm text-[#8A8A8A] cursor-pointer">
                    <input
                      type="checkbox"
                      checked={castling.k}
                      onChange={() => toggleCastling('k')}
                      className="accent-[#6B9E6B]"
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
                      className="accent-[#6B9E6B]"
                    />
                    Queenside (O-O-O)
                  </label>
                )}
              </div>
            )}
          </div>
        </section>
      )}

      {/* FEN display */}
      <div className="w-full max-w-lg bg-[#242424] rounded-xl border border-[#333] p-4">
        <p className="text-[#8A8A8A] text-xs uppercase tracking-wider mb-2">FEN</p>
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
