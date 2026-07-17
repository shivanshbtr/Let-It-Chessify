import { useState } from 'react'
import { ChevronRight, ArrowLeft } from 'lucide-react'

export default function TurnSelectStep({ fen, onConfirm, onBack }) {
  // Initialize from the FEN's own side-to-move field, so coming back to this
  // step (e.g. from Analysis) shows whatever was previously selected instead
  // of always resetting to White.
  const [turn, setTurn] = useState(() => (fen.split(' ')[1] === 'b' ? 'b' : 'w'))

  // Update FEN with selected turn
  const getFenWithTurn = (t) => {
    const parts = fen.split(' ')
    parts[1] = t
    return parts.join(' ')
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-full gap-8 px-4 py-12 relative">
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
        <h2 className="font-display text-3xl font-semibold text-[#F5F0E8] mb-2">
          Who moves next?
        </h2>
        <p className="text-[#8A8A8A] text-sm">
          Select the side to move before analysis begins.
        </p>
      </div>

      {/* Turn selector */}
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
