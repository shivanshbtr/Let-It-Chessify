import { useState, useCallback, useEffect } from 'react'
import { Chessboard, ChessboardProvider, SparePiece } from 'react-chessboard'
import { RotateCcw, FlipVertical, Check, AlertTriangle, ArrowLeft } from 'lucide-react'
import { editFen } from '../api/chess'
import { useSquareFit } from '../hooks/useSquareFit'

const PIECES = ['wK','wQ','wR','wB','wN','wP','bK','bQ','bR','bB','bN','bP']

function labelsToBoardPosition(squareLabels) {
  const pos = {}
  Object.entries(squareLabels).forEach(([sq, label]) => {
    if (label !== 'Empty') pos[sq] = { pieceType: label }
  })
  return pos
}

function flip180(squareLabels) {
  const files = ['a','b','c','d','e','f','g','h']
  const ranks = ['1','2','3','4','5','6','7','8']
  const flipped = {}
  files.forEach((f, fi) => {
    ranks.forEach((r, ri) => {
      const src = `${f}${r}`
      const dstFile = files[7 - fi]
      const dstRank = ranks[7 - ri]
      const dst = `${dstFile}${dstRank}`
      flipped[dst] = squareLabels[src] || 'Empty'
    })
  })
  return flipped
}

// Crop the original photo down to just the detected board (bounding box of the
// confirmed corners, plus a small margin) so it can be shown for verification.
function cropToCorners(originalB64, corners) {
  return new Promise((resolve, reject) => {
    if (!originalB64 || !corners || corners.length !== 4) {
      resolve(null)
      return
    }
    const img = new Image()
    img.onload = () => {
      const xs = corners.map(c => c[0])
      const ys = corners.map(c => c[1])
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs)
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys)

      const w = maxX - minX
      const h = maxY - minY
      const marginX = w * 0.04
      const marginY = h * 0.04

      const sx = Math.max(0, minX - marginX)
      const sy = Math.max(0, minY - marginY)
      const sw = Math.min(img.naturalWidth - sx, w + marginX * 2)
      const sh = Math.min(img.naturalHeight - sy, h + marginY * 2)

      const canvas = document.createElement('canvas')
      canvas.width = sw
      canvas.height = sh
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh)
      resolve(canvas.toDataURL('image/png'))
    }
    img.onerror = reject
    img.src = `data:image/png;base64,${originalB64}`
  })
}

export default function BoardEditorStep({
  squareLabels: initialLabels,
  warnings: initialWarnings,
  originalB64,
  corners,
  onConfirm,
  onBack,
}) {
  const [labels, setLabels]         = useState(initialLabels)
  const [warnings, setWarnings]     = useState(initialWarnings)
  const [saving, setSaving]         = useState(false)
  const [croppedUrl, setCroppedUrl] = useState(null)
  const [boardFitRef, boardSize]    = useSquareFit()

  useEffect(() => {
    let cancelled = false
    cropToCorners(originalB64, corners)
      .then(url => { if (!cancelled) setCroppedUrl(url) })
      .catch(err => console.error('Crop failed:', err))
    return () => { cancelled = true }
  }, [originalB64, corners])

  const position = labelsToBoardPosition(labels)

  // Drag piece → move it. Covers board-to-board moves, dragging a piece in
  // from the palette (a "spare piece"), and dragging a piece off the board
  // entirely to delete it.
  //
  // For a spare piece, sourceSquare is actually just the piece type string
  // (e.g. "wK"), not a real board square, so it must NOT be cleared like a
  // normal move would.
  const onPieceDrop = useCallback(({ piece, sourceSquare, targetSquare }) => {
    if (!targetSquare) {
      // Dropped outside the board entirely. For an existing board piece,
      // that's how you delete it -- remove it from its square. A spare
      // piece dropped off-board was never placed, so it's a no-op.
      if (!piece.isSparePiece) {
        setLabels(prev => ({ ...prev, [sourceSquare]: 'Empty' }))
      }
      return false
    }
    const newLabels = { ...labels }
    if (!piece.isSparePiece) {
      newLabels[sourceSquare] = 'Empty'
    }
    newLabels[targetSquare] = piece.pieceType
    setLabels(newLabels)
    return true
  }, [labels])

  // Right-click square → delete piece
  const onSquareRightClick = useCallback(({ square }) => {
    const newLabels = { ...labels, [square]: 'Empty' }
    setLabels(newLabels)
  }, [labels])

  // Flip board 180° (fixes orientation if camera was from wrong side)
  const handleFlip = () => {
    setLabels(flip180(labels))
  }

  const handleConfirm = async () => {
    setSaving(true)
    try {
      // Validate via backend
      const res = await editFen(labels, 'w')
      onConfirm(labels, res.fen, res.warnings || [])
    } catch (e) {
      console.error(e)
    }
    setSaving(false)
  }

  const chessboardOptions = {
    position,
    onPieceDrop,
    onSquareRightClick,
    boardOrientation: 'white',
    allowDrawingArrows: false,
    boardStyle: { borderRadius: '8px', boxShadow: '0 4px 24px rgba(0,0,0,0.4)' },
    darkSquareStyle: { backgroundColor: '#4A3728' },
    lightSquareStyle: { backgroundColor: '#F5F0E8' },
  }

  return (
    <div className="flex flex-col gap-4 h-full px-4 py-6">
      <div>
        <div className="flex items-center justify-between">
          <h2 className="font-display text-2xl font-semibold text-[#F5F0E8]">
            Review detected position
          </h2>
          {onBack && (
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 text-[#8A8A8A] hover:text-[#F5F0E8]
                         text-sm transition-colors"
            >
              <ArrowLeft size={14} />
              Back
            </button>
          )}
        </div>
        <p className="text-[#8A8A8A] text-sm mt-1">
          Drag pieces to move · Right-click to remove · Drag from palette to add
        </p>
        <p className="text-[#8A8A8A]/70 text-xs mt-1">
          If the black pieces are at the bottom of your original photo, click{' '}
          <span className="text-[#F5F0E8]">Flip</span> to correct the orientation.
        </p>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {warnings.map((w, i) => (
            <div key={i} className="flex items-center gap-2 bg-[#C8A96E]/10 border border-[#C8A96E]/30
                                    rounded-lg px-3 py-2 text-[#C8A96E] text-xs">
              <AlertTriangle size={13} />
              {w}
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Reference crop: the detected board area from the original photo, for side-by-side verification */}
        {croppedUrl && (
          <div className="flex flex-col gap-2 w-64 flex-shrink-0">
            <p className="text-[#8A8A8A] text-[10px] uppercase tracking-wider text-center">
              Detected board (reference)
            </p>
            <div className="rounded-xl overflow-hidden border border-[#333] bg-[#111]">
              <img
                src={croppedUrl}
                alt="Cropped detected board"
                className="w-full h-auto object-contain"
                draggable={false}
              />
            </div>
          </div>
        )}

        {/* Board + palette share one ChessboardProvider so dragging a palette
            piece onto the board works -- SparePiece and Chessboard must live
            in the same DnD context to talk to each other. */}
        <ChessboardProvider options={chessboardOptions}>
          {/* Board */}
          <div ref={boardFitRef} className="flex-1 min-w-0 h-full flex items-center justify-center">
            {boardSize > 0 && (
              <div style={{ width: boardSize, height: boardSize }}>
                <Chessboard />
              </div>
            )}
          </div>

          {/* Right panel: palette + actions */}
          <div className="flex flex-col gap-4 w-24">
            {/* Piece palette */}
            <div className="bg-[#242424] rounded-xl border border-[#333] p-2">
              <p className="text-[#8A8A8A] text-[10px] uppercase tracking-wider mb-2 text-center">
                Drag to add
              </p>
              <div className="grid grid-cols-2 gap-1.5">
                {PIECES.map(p => {
                  const isBlack = p.startsWith('b')
                  return (
                    <div
                      key={p}
                      title={p}
                      className={`palette-piece flex items-center justify-center rounded-lg py-1.5 transition-all ${
                        isBlack
                          ? 'bg-[#F5F0E8] hover:bg-[#e8e2d5]'
                          : 'bg-[#1A1A1A] hover:bg-[#333]'
                      }`}
                    >
                      <div style={{ width: 28, height: 28 }}>
                        <SparePiece pieceType={p} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Board controls */}
            <div className="flex flex-col gap-2">
              <button
                onClick={handleFlip}
                title="Use if black pieces are at the bottom of your photo"
                className="flex flex-col items-center gap-1 px-2 py-2.5 rounded-lg
                           border border-[#333] text-[#8A8A8A] hover:text-[#F5F0E8]
                           hover:border-[#555] transition-all text-xs"
              >
                <FlipVertical size={16} />
                Flip
              </button>
              <button
                onClick={() => setLabels(initialLabels)}
                className="flex flex-col items-center gap-1 px-2 py-2.5 rounded-lg
                           border border-[#333] text-[#8A8A8A] hover:text-[#F5F0E8]
                           hover:border-[#555] transition-all text-xs"
              >
                <RotateCcw size={16} />
                Reset
              </button>
            </div>
          </div>
        </ChessboardProvider>
      </div>

      {/* Confirm */}
      <button
        onClick={handleConfirm}
        disabled={saving}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl
                   bg-[#6B9E6B] hover:bg-[#7aaf7a] text-white font-medium
                   transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {saving ? (
          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : (
          <><Check size={16} /> Position looks correct</>
        )}
      </button>
    </div>
  )
}
