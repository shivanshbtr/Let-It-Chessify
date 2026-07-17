import { useState, useCallback } from 'react'
import { Chess } from 'chess.js'
import StepIndicator   from './components/StepIndicator'
import UploadStep      from './components/UploadStep'
import CornerConfirmStep from './components/CornerConfirmStep'
import BoardEditorStep from './components/BoardEditorStep'
import TurnSelectStep  from './components/TurnSelectStep'
import AnalysisStep    from './components/AnalysisStep'
import { detectCorners, classify } from './api/chess'

const STEP = { UPLOAD: 1, CORNERS: 2, EDITOR: 3, TURN: 4, ANALYSIS: 5 }

// Convert a FEN into the { square: 'wP' | 'bK' | ... | 'Empty' } shape the
// board editor works with -- used when jumping straight to analysis from
// the standard starting position, with no photo/detection involved at all.
function fenToSquareLabels(fen) {
  const g = new Chess(fen)
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
  const labels = {}
  g.board().forEach((row, rIdx) => {
    const rank = 8 - rIdx
    row.forEach((cell, cIdx) => {
      const square = `${files[cIdx]}${rank}`
      labels[square] = cell ? `${cell.color}${cell.type.toUpperCase()}` : 'Empty'
    })
  })
  return labels
}

// Parse a PGN string into the timeline shape AnalysisStep works with:
// the starting FEN (respecting a [FEN]/[SetUp] header for non-standard
// starts), the full list of position FENs after each ply, and the SAN
// for each ply. Throws if the PGN has no legal moves chess.js can parse.
function parsePgn(pgnText) {
  const g = new Chess()
  g.loadPgn(pgnText, { strict: false })
  const verboseMoves = g.history({ verbose: true })
  const startFen = verboseMoves.length ? verboseMoves[0].before : g.fen()
  const positionHistory = [startFen]
  const sanHistory = []
  verboseMoves.forEach((mv) => {
    sanHistory.push(mv.san)
    positionHistory.push(mv.after)
  })
  const startTurn = new Chess(startFen).turn()
  return { startFen, positionHistory, sanHistory, startTurn }
}

export default function App() {
  const [step, setStep]               = useState(STEP.UPLOAD)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [overlayB64, setOverlayB64]   = useState(null)
  const [originalB64, setOriginalB64] = useState(null)
  const [corners, setCorners]         = useState(null)
  const [confirmedCorners, setConfirmedCorners] = useState(null)
  const [grid, setGrid]               = useState(null)
  const [isPhysical, setIsPhysical]   = useState(false)
  const [squareLabels, setSquareLabels] = useState(null)
  const [warnings, setWarnings]         = useState([])
  const [confirmedFen, setConfirmedFen] = useState(null)
  const [analysisData, setAnalysisData] = useState(null)

  const clearError = () => setError(null)

  // Which steps the user can actually jump to right now, based on what data
  // exists -- not just "have they scrolled past it before". This is what
  // lets the step indicator and Back buttons offer real, non-linear
  // navigation (e.g. Analysis -> Editor to fix a piece) instead of a rigid
  // one-way wizard, while still refusing to jump into a step that has
  // nothing to show (e.g. Corners when there's no photo at all).
  const availableSteps = [
    STEP.UPLOAD,
    ...(overlayB64 && originalB64 ? [STEP.CORNERS] : []),
    ...(squareLabels ? [STEP.EDITOR] : []),
    ...(confirmedFen ? [STEP.TURN] : []),
    ...(analysisData ? [STEP.ANALYSIS] : []),
  ]

  const goToStep = useCallback((target) => {
    clearError()
    setStep(target)
  }, [])

  const handleUpload = useCallback(async (file, physical) => {
    clearError()
    setLoading(true)
    setIsPhysical(physical)
    try {
      const res = await detectCorners(file, physical)
      if (!res.success) {
        setError(res.message || 'Board detection failed.')
        setLoading(false)
        return
      }
      setOverlayB64(res.overlay_image_b64)
      setOriginalB64(res.original_image_b64)
      setCorners(res.corners)
      setGrid(res.grid)
      setStep(STEP.CORNERS)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }, [])

  const handleCornersConfirm = useCallback(async (confirmedGrid, cornersAtConfirm) => {
    clearError()
    setLoading(true)
    try {
      const res = await classify(originalB64, confirmedGrid, isPhysical)
      if (!res.success) {
        setError(res.message || 'Classification failed.')
        setLoading(false)
        return
      }
      setSquareLabels(res.square_labels)
      setWarnings(res.warnings || [])
      // Remember the grid/corners as last-confirmed, so if the user later
      // comes back to this step (e.g. from the editor, to fix a corner),
      // they see what they actually confirmed rather than the original
      // raw detection guess.
      setCorners(cornersAtConfirm)
      setGrid(confirmedGrid)
      setConfirmedCorners(cornersAtConfirm)
      setStep(STEP.EDITOR)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }, [originalB64, isPhysical])

  const handleEditorConfirm = useCallback((labels, fen, warns) => {
    // Persist the user's edits, so coming back to the editor later (from
    // Turn or Analysis) shows the position as they left it, not the
    // original detection.
    setSquareLabels(labels)
    setConfirmedFen(fen)
    setWarnings(warns)
    setStep(STEP.TURN)
  }, [])

  const handleTurnConfirm = useCallback((turn, fenWithTurn) => {
    setConfirmedFen(fenWithTurn)
    setAnalysisData({ fen: fenWithTurn, turn })
    setStep(STEP.ANALYSIS)
  }, [])

  // Skip photo/detection entirely and jump straight to analysing the
  // standard starting position. Still populates squareLabels/confirmedFen
  // so the Editor and Turn steps stay reachable afterwards if the person
  // wants to set up a custom position instead of the default.
  const handleStartFromScratch = useCallback(() => {
    clearError()
    const startFen = new Chess().fen()
    setSquareLabels(fenToSquareLabels(startFen))
    setWarnings([])
    setConfirmedFen(startFen)
    setAnalysisData({ fen: startFen, turn: 'w' })
    setStep(STEP.ANALYSIS)
  }, [])

  // Import a PGN and jump straight to analysis with the whole game loaded
  // as move history (not just the final position) -- mirrors
  // handleStartFromScratch in populating squareLabels/confirmedFen so the
  // Editor and Turn steps stay reachable afterwards, but additionally
  // carries the parsed move timeline through to AnalysisStep so Back/
  // Forward can step through the imported game.
  const handleImportPgn = useCallback((pgnText) => {
    clearError()
    try {
      const { startFen, positionHistory, sanHistory, startTurn } = parsePgn(pgnText)
      setSquareLabels(fenToSquareLabels(startFen))
      setWarnings([])
      setConfirmedFen(startFen)
      setAnalysisData({
        fen: startFen,
        turn: startTurn === 'w' ? 'w' : 'b',
        initialHistory: { positionHistory, sanHistory },
      })
      setStep(STEP.ANALYSIS)
    } catch (e) {
      setError('Could not read that PGN. Double-check the format and try again.')
    }
  }, [])

  const handleReset = useCallback(() => {
    setStep(STEP.UPLOAD)
    setOverlayB64(null); setOriginalB64(null)
    setCorners(null); setGrid(null); setConfirmedCorners(null)
    setSquareLabels(null); setWarnings([])
    setConfirmedFen(null); setAnalysisData(null)
    setError(null); setLoading(false)
  }, [])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#1A1A1A]">
      <div className="border-r border-[#242424] flex-shrink-0">
        <StepIndicator
          currentStep={step}
          availableSteps={availableSteps}
          onStepClick={goToStep}
        />
      </div>
      <div className="flex-1 overflow-y-auto relative">
        {error && (
          <div className="absolute top-4 left-4 right-4 z-50 bg-red-900/80 border
                          border-red-700 text-red-200 text-sm px-4 py-3 rounded-xl
                          flex items-center justify-between backdrop-blur">
            <span>{error}</span>
            <button
              onClick={() => { clearError(); if (step > STEP.UPLOAD) setStep(STEP.UPLOAD) }}
              className="ml-4 text-red-300 hover:text-white underline text-xs flex-shrink-0"
            >
              Try again
            </button>
          </div>
        )}
        <div className="h-full">
          {step === STEP.UPLOAD && (
            <UploadStep
              onUpload={handleUpload}
              onStartFromScratch={handleStartFromScratch}
              onImportPgn={handleImportPgn}
              loading={loading}
            />
          )}
          {step === STEP.CORNERS && overlayB64 && (
            <CornerConfirmStep
              overlayB64={overlayB64}
              originalB64={originalB64}
              initialCorners={corners}
              initialGrid={grid}
              isPhysical={isPhysical}
              onConfirm={handleCornersConfirm}
              onBack={() => goToStep(STEP.UPLOAD)}
              loading={loading}
            />
          )}
          {step === STEP.EDITOR && squareLabels && (
            <BoardEditorStep
              squareLabels={squareLabels}
              warnings={warnings}
              originalB64={originalB64}
              corners={confirmedCorners}
              onConfirm={handleEditorConfirm}
              onBack={availableSteps.includes(STEP.CORNERS) ? () => goToStep(STEP.CORNERS) : null}
            />
          )}
          {step === STEP.TURN && confirmedFen && (
            <TurnSelectStep
              fen={confirmedFen}
              onConfirm={handleTurnConfirm}
              onBack={() => goToStep(STEP.EDITOR)}
            />
          )}
          {step === STEP.ANALYSIS && analysisData && (
            <AnalysisStep
              fen={analysisData.fen}
              turn={analysisData.turn}
              initialHistory={analysisData.initialHistory}
              onReset={handleReset}
              onBack={() => goToStep(STEP.TURN)}
            />
          )}
        </div>
      </div>
    </div>
  )
}
