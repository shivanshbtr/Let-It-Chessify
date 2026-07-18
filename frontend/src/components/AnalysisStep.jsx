import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Chessboard, ChessboardProvider } from 'react-chessboard'
import { Chess } from 'chess.js'
import { RotateCcw, RefreshCw, ArrowLeft, Undo2, Redo2, FlipVertical, Download } from 'lucide-react'
import ScoreBar from './ScoreBar'
import { analyzeStream } from '../api/chess'
import { useSquareFit } from '../hooks/useSquareFit'

// Arrow colors: green=best, blue=second, yellow=third
const ARROW_COLORS = ['#00C853', '#2979FF', '#FF6D00']
// On-board arrow opacity: best move stays fully solid, 2nd/3rd fade back
// so they don't compete visually with the top suggestion. Gap between 1st
// and 2nd is kept wide so the top pick reads unambiguously; 2nd/3rd stay
// closer together since they're both "alternatives" rather than "the move".
const ARROW_OPACITIES = [1, 0.75, 0.55]

// Combine an arrow's base hex color with its opacity into an rgba() string
function arrowColor(hex, opacity) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${opacity})`
}
const MODE_SCORE = 'score'
const MODE_SUGGEST = 'suggest'

// Normalize the detected starting position into a canonical FEN string.
function normalizeInitialFen(initialFen) {
  const g = new Chess()
  try { g.load(initialFen) } catch {}
  return g.fen()
}

export default function AnalysisStep({ fen: initialFen, turn, initialHistory, onReset, onBack }) {
  const [mode, setMode]         = useState(MODE_SCORE)
  const [evalData, setEvalData] = useState(null)
  const [loading, setLoading]   = useState(false)
  const [boardOrientation, setBoardOrientation] = useState(turn === 'w' ? 'white' : 'black')

  // Position/move history is modeled as a single timeline with a cursor:
  //   positionHistory[0]      = starting position (FEN)
  //   positionHistory[i]      = position after sanHistory[i-1] was played
  //   currentIndex            = which position is currently shown on the board
  // Back/Undo moves currentIndex left, Forward/Redo moves it right -- neither
  // one destroys the timeline, so you can freely go back and then forward
  // again. Playing a new move while currentIndex is behind the tip discards
  // the "future" beyond it and branches from there, same as any chess UI.
  //
  // When arriving from a PGN import, initialHistory carries the whole parsed
  // game so the timeline starts pre-populated -- the board opens on the
  // final position of the import with the full game browsable via
  // Back/Forward, rather than dropping the imported moves on the floor.
  const [positionHistory, setPositionHistory] = useState(() =>
    initialHistory?.positionHistory?.length
      ? initialHistory.positionHistory
      : [normalizeInitialFen(initialFen)]
  )
  const [sanHistory, setSanHistory]     = useState(() => initialHistory?.sanHistory ?? [])
  const [currentIndex, setCurrentIndex] = useState(() =>
    initialHistory?.positionHistory?.length ? initialHistory.positionHistory.length - 1 : 0
  )

  const analysisRequestId = useRef(0)
  const [boardFitRef, boardSize] = useSquareFit()

  const game = useMemo(() => {
    const g = new Chess()
    try { g.load(positionHistory[currentIndex]) } catch {}
    return g
  }, [positionHistory, currentIndex])

  const currentFen = game.fen()
  const canGoBack = currentIndex > 0
  const canGoForward = currentIndex < positionHistory.length - 1

  const closeStreamRef = useRef(null)

  const fetchEval = useCallback((fen, currentTurn) => {
    const requestId = ++analysisRequestId.current

    // Close any in-flight stream from a previous position before starting
    // a new one, so two analyses never race to update the UI at once.
    closeStreamRef.current?.()

    setLoading(true)

    closeStreamRef.current = analyzeStream(
      fen,
      currentTurn,
      3,
      (update) => {
        if (requestId !== analysisRequestId.current) return
        setEvalData(update)
        if (update.done) setLoading(false)
      },
      (e) => {
        if (requestId !== analysisRequestId.current) return
        console.error('Analysis failed:', e)
        setLoading(false)
      }
    )
  }, [])

  useEffect(() => {
    return () => closeStreamRef.current?.()
  }, [])

  // Fetch eval whenever position changes -- debounced so that spamming
  // Back/Forward (or holding the arrow key) doesn't fire one analyze()
  // call per intermediate position. Without this, a fast run of clicks
  // queues a burst of requests that resolve out of order; even though
  // fetchEval's requestId guard stops stale ones from overwriting the
  // final result, the score bar/arrows still visibly flicker through
  // whichever intermediate evals happen to land first. Waiting for the
  // position to sit still for a beat means only the position the user
  // actually stops on ever gets analysed.
  useEffect(() => {
    const t = game.turn()
    const timeoutId = setTimeout(() => {
      fetchEval(currentFen, t)
    }, 300)
    return () => clearTimeout(timeoutId)
  }, [currentFen, fetchEval])

  // Apply a move to the current position. Used both for dragging pieces on
  // the board and for clicking a suggested move -- the board should always
  // be playable, in either mode. If we're not at the tip of the timeline
  // (the user went Back first), this branches: anything after currentIndex
  // is dropped and replaced by the new move.
  const makeMove = useCallback(({ from, to, promotion = 'q' }) => {
    try {
      const newGame = new Chess(positionHistory[currentIndex])
      const move = newGame.move({ from, to, promotion })
      if (!move) return false
      setPositionHistory(ph => [...ph.slice(0, currentIndex + 1), newGame.fen()])
      setSanHistory(sh => [...sh.slice(0, currentIndex), move.san])
      setCurrentIndex(i => i + 1)
      return true
    } catch {
      return false
    }
  }, [positionHistory, currentIndex])

  // v5 callback signature: ({ piece, sourceSquare, targetSquare }) => boolean
  const onPieceDrop = useCallback(({ sourceSquare, targetSquare }) => {
    if (!targetSquare) return false
    return makeMove({ from: sourceSquare, to: targetSquare })
  }, [makeMove])

  // Move suggestions mode: clicking a suggestion plays it
  const playSuggestion = (uci) => {
    makeMove({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' })
  }

  // Flip the board view only -- this is a display preference, so it doesn't
  // touch the position/move data at all (unlike the board editor's flip,
  // which corrects mislabeled squares from a wrong-side photo).
  const handleFlip = useCallback(() => {
    setBoardOrientation(o => (o === 'white' ? 'black' : 'white'))
  }, [])

  const handleReset = useCallback(() => {
    const fen = normalizeInitialFen(initialFen)
    setPositionHistory([fen])
    setSanHistory([])
    setCurrentIndex(0)
  }, [initialFen])

  // Export the full move timeline (not just the currently-viewed position)
  // as a downloadable .pgn file. Replays sanHistory from positionHistory[0]
  // on a scratch Chess instance so chess.js can generate proper PGN move
  // numbering/headers -- including a [FEN]/[SetUp] header automatically
  // when the game didn't start from the standard position.
  const exportPgn = useCallback(() => {
    try {
      const g = new Chess(positionHistory[0])
      sanHistory.forEach((san) => { g.move(san) })
      const pgn = g.pgn()
      const blob = new Blob([pgn], { type: 'application/x-chess-pgn' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'chessify-game.pgn'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error('PGN export failed:', e)
    }
  }, [positionHistory, sanHistory])

  // Take back the last move. Repeatable -- each call steps back one ply,
  // all the way to the original detected position.
  const handleUndo = useCallback(() => {
    setCurrentIndex(i => (i > 0 ? i - 1 : i))
  }, [])

  // Step forward again through moves that were undone. Repeatable -- each
  // call steps forward one ply, up to the most recently played move.
  const handleRedo = useCallback(() => {
    setCurrentIndex(i => (i < positionHistory.length - 1 ? i + 1 : i))
  }, [positionHistory.length])

  // Left/Right arrow keys step back and forward through the move timeline,
  // same as the Back/Forward buttons. Ignored while typing in a field.
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = document.activeElement?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || document.activeElement?.isContentEditable) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        handleUndo()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        handleRedo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [handleUndo, handleRedo])

  // Build arrows for suggestion mode
  // v5 arrow shape: { startSquare, endSquare, color }
  const arrows = (mode === MODE_SUGGEST && evalData?.best_moves)
    ? evalData.best_moves.slice(0, 3).map((mv, i) => ({
        startSquare: mv.uci.slice(0, 2),
        endSquare:   mv.uci.slice(2, 4),
        color:       arrowColor(ARROW_COLORS[i], ARROW_OPACITIES[i]),
      }))
    : []

  const evalCp   = evalData?.eval_cp   ?? null
  const depth    = evalData?.depth     ?? null
  const maxDepth = 40
  const evalType = evalData?.eval_type ?? 'cp'
  const mateIn   = evalData?.mate_in   ?? null

  // Moves shown in the move-list reflect the currently viewed position,
  // i.e. everything up to currentIndex -- not the full (possibly longer)
  // timeline if the user has stepped back.
  const visibleMoves = sanHistory.slice(0, currentIndex)

  return (
    <div className="flex flex-col h-full px-4 py-6 gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-display text-2xl font-semibold text-[#F5F0E8]">
          Analysis
        </h2>
        <div className="flex items-center gap-4">
          {onBack && (
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 text-[#8A8A8A] hover:text-[#F5F0E8]
                         text-sm transition-colors"
            >
              <Undo2 size={14} />
              Back to turn
            </button>
          )}
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 text-[#8A8A8A] hover:text-[#F5F0E8]
                       text-sm transition-colors"
          >
            <ArrowLeft size={14} />
            New position
          </button>
        </div>
      </div>

      {/* Mode toggle */}
      <div className="flex bg-[#242424] rounded-full p-1 border border-[#333]">
        <button
          onClick={() => setMode(MODE_SCORE)}
          className={`flex-1 text-sm py-1.5 rounded-full transition-all font-medium ${
            mode === MODE_SCORE
              ? 'bg-[#6B9E6B] text-white'
              : 'text-[#8A8A8A] hover:text-[#F5F0E8]'
          }`}
        >
          Score bar
        </button>
        <button
          onClick={() => setMode(MODE_SUGGEST)}
          className={`flex-1 text-sm py-1.5 rounded-full transition-all font-medium ${
            mode === MODE_SUGGEST
              ? 'bg-[#6B9E6B] text-white'
              : 'text-[#8A8A8A] hover:text-[#F5F0E8]'
          }`}
        >
          Move suggestions
        </button>
      </div>

      {/* Move suggestions lines (above board) */}
      {mode === MODE_SUGGEST && evalData?.best_moves && (
        <div className="flex items-center gap-3 px-2 py-1.5 rounded-lg bg-[#242424]
                        border border-[#333] text-xs">
          {evalData.best_moves.slice(0, 3).map((mv, i) => {
            const evalStr =
              mv.eval_type === 'mate'
                ? `M${mv.mate_in ?? '?'}`
                : mv.eval_cp !== null
                  ? (mv.eval_cp > 0 ? '+' : '') + (mv.eval_cp / 100).toFixed(1)
                  : '?'

            return (
              <button
                key={i}
                onClick={() => playSuggestion(mv.uci)}
                className="flex items-center gap-1.5 hover:opacity-80"
              >
                {/* Original color dot */}
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: ARROW_COLORS[i] }}
                />

                {/* Label */}
                <span
                  className="font-semibold"
                  style={{ color: ARROW_COLORS[i] }}
                >
                  {i === 0 ? 'Best' : i === 1 ? '2nd' : '3rd'}
                </span>

                {/* Move */}
                <span className="text-[#F5F0E8] font-mono">
                  {mv.san || mv.uci}
                </span>

                {/* Eval */}
                <span
                  className="font-mono"
                  style={{ color: ARROW_COLORS[i] }}
                >
                  {evalStr}
                </span>

                {i < 2 && <span className="text-[#555] ml-1">|</span>}
              </button>
            )
          })}
        </div>
      )}

      {/* Board + score bar */}
      <div className="flex gap-3 flex-1 min-h-0 items-start">
        <div ref={boardFitRef} className="flex-1 min-w-0 h-full flex items-center justify-center">
          {boardSize > 0 && (
            <div style={{ width: boardSize, height: boardSize }}>
              <ChessboardProvider
                options={{
                  position: game.fen(),
                  onPieceDrop,
                  boardOrientation,
                  allowDrawingArrows: false,
                  arrows,
                  boardStyle: {
                    borderRadius: '8px',
                    boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
                  },
                  darkSquareStyle: { backgroundColor: '#4A3728' },
                  lightSquareStyle: { backgroundColor: '#F5F0E8' },
                }}
              >
                <Chessboard />
              </ChessboardProvider>
            </div>
          )}
        </div>

        {/* Score bar (always visible) */}
        <div className="h-full flex flex-col items-center" style={{ minHeight: 200 }}>
          {!evalData ? (
            <div className="w-8 h-full flex items-center justify-center">
              <div className="w-3 h-3 border border-[#6B9E6B] border-t-transparent
                              rounded-full animate-spin" />
            </div>
          ) : (
            <>
              <ScoreBar evalCp={evalCp} evalType={evalType} mateIn={mateIn} />
              {loading && depth != null && (
                <span className="text-[8px] text-[#8A8A8A] mt-1 whitespace-nowrap">
                  d{depth}/{maxDepth}
                </span>
              )}
            </>
          )}
        </div>
      </div>

      {/* Move history + controls */}
      <div className="flex items-center gap-3">
        {/* Move history */}
        <div className="flex-1 flex flex-wrap gap-1 min-h-6">
          {visibleMoves.map((mv, i) => (
            <span key={i} className="text-xs text-[#8A8A8A] font-mono">
              {i % 2 === 0 && (
                <span className="text-[#555] mr-0.5">{Math.floor(i/2)+1}.</span>
              )}
              {mv}
            </span>
          ))}
        </div>

        {/* Undo last move -- repeatable, one ply per click (or ←) */}
        <button
          onClick={handleUndo}
          disabled={!canGoBack}
          title="Back (←)"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0 disabled:opacity-40
                     disabled:cursor-not-allowed"
        >
          <Undo2 size={12} />
          Back
        </button>

        {/* Redo/step forward -- repeatable, one ply per click (or →) */}
        <button
          onClick={handleRedo}
          disabled={!canGoForward}
          title="Forward (→)"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0 disabled:opacity-40
                     disabled:cursor-not-allowed"
        >
          <Redo2 size={12} />
          Forward
        </button>

        {/* Flip board view */}
        <button
          onClick={handleFlip}
          title="Flip board"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0"
        >
          <FlipVertical size={12} />
          Flip
        </button>

        {/* Reset to detected position */}
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0"
        >
          <RotateCcw size={12} />
          Reset position
        </button>

        {/* Re-analyse */}
        <button
          onClick={() => fetchEval(currentFen, game.turn())}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0 disabled:opacity-40"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Re-analyse
        </button>

        {/* Export the full game as a .pgn file */}
        <button
          onClick={exportPgn}
          disabled={sanHistory.length === 0}
          title="Export PGN"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555]
                     transition-all text-xs flex-shrink-0 disabled:opacity-40
                     disabled:cursor-not-allowed"
        >
          <Download size={12} />
          Export PGN
        </button>
      </div>

      {/* Mode hint */}
      <p className="text-[#555] text-xs">
        {mode === MODE_SCORE
          ? 'Play any move on the board — score bar updates after each move'
          : 'Green = best · Blue = second · Yellow = third — play any move, or click a suggestion'}
      </p>
    </div>
  )
}
