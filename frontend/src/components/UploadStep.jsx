import { useState, useCallback, useRef, useEffect } from 'react'
import { Upload, Camera, MonitorSmartphone, Swords, FileUp, FolderOpen, X } from 'lucide-react'

export default function UploadStep({ onUpload, onStartFromScratch, onImportPgn, loading }) {
  const [isDragging, setIsDragging] = useState(false)
  const [isPhysical, setIsPhysical] = useState(false)
  const [showPgnPanel, setShowPgnPanel] = useState(false)
  const [pgnText, setPgnText] = useState('')
  const pgnFileInputRef = useRef(null)
  const pgnPanelRef = useRef(null)
  const pgnTextareaRef = useRef(null)

  // Scroll the paste box into view (and focus it) the moment it opens --
  // the panel renders below the fold on shorter viewports, so without this
  // clicking "Import PGN" looks like nothing happened.
  useEffect(() => {
    if (!showPgnPanel) return
    const raf = requestAnimationFrame(() => {
      pgnPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      pgnTextareaRef.current?.focus()
    })
    return () => cancelAnimationFrame(raf)
  }, [showPgnPanel])

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    onUpload(file, isPhysical)
  }, [onUpload, isPhysical])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = () => setIsDragging(false)
  const onFileInput = (e) => handleFile(e.target.files[0])

  const submitPgn = useCallback(() => {
    if (!pgnText.trim() || !onImportPgn) return
    onImportPgn(pgnText)
  }, [pgnText, onImportPgn])

  const onPgnFileInput = useCallback((e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const text = String(reader.result || '')
      setPgnText(text)
      if (onImportPgn) onImportPgn(text)
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [onImportPgn])

  return (
    <div className="relative min-h-full overflow-hidden" style={{ background: 'var(--walnut-deep)' }}>
      {/* Ambient dressing: faint checkerboard veil + warm vignette + oversized watermark glyph */}
      <div className="absolute inset-0 checker-veil" />
      <div className="absolute inset-0 hero-vignette" />
      <span
        className="watermark-glyph absolute -top-12 right-[6%] text-[22rem] leading-none select-none hidden md:block"
        aria-hidden="true"
      >
        ♛
      </span>

      {/* Side strip: quick-access options, pinned to the right edge and
          higher up (near the title) instead of buried below the drop
          zone where they were easy to miss / needed scrolling to reach. */}
      <div className="hidden sm:flex absolute right-5 top-24 z-20 flex-col items-center gap-6">
        {onStartFromScratch && (
          <button
            onClick={onStartFromScratch}
            disabled={loading}
            title="Analyse from the starting position"
            className="flex flex-col items-center gap-1.5 group
                       disabled:opacity-40 disabled:pointer-events-none"
          >
            <span
              className="w-10 h-10 rounded-full flex items-center justify-center
                         transition-transform duration-150 group-hover:scale-110"
              style={{ background: 'var(--walnut-raised)', border: '1px solid var(--hairline)' }}
            >
              <Swords size={16} style={{ color: 'var(--brass)' }} />
            </span>
            <span
              className="text-[9px] uppercase tracking-wider text-center leading-tight max-w-[68px]"
              style={{ color: 'var(--stone)' }}
            >
              Start position
            </span>
          </button>
        )}
        {onImportPgn && (
          <button
            onClick={() => setShowPgnPanel((v) => !v)}
            disabled={loading}
            title="Import PGN"
            className="flex flex-col items-center gap-1.5 group
                       disabled:opacity-40 disabled:pointer-events-none"
          >
            <span
              className="w-10 h-10 rounded-full flex items-center justify-center
                         transition-transform duration-150 group-hover:scale-110"
              style={{
                background: showPgnPanel ? 'var(--felt)' : 'var(--walnut-raised)',
                border: `1px solid ${showPgnPanel ? 'var(--felt-bright)' : 'var(--hairline)'}`,
              }}
            >
              <FileUp size={16} style={{ color: showPgnPanel ? 'var(--ivory)' : 'var(--brass)' }} />
            </span>
            <span
              className="text-[9px] uppercase tracking-wider text-center leading-tight max-w-[68px]"
              style={{ color: 'var(--stone)' }}
            >
              Import PGN
            </span>
          </button>
        )}
      </div>

      <div className="relative flex flex-col items-center justify-center min-h-full gap-9 px-4 py-16">
        {/* Eyebrow */}
        <div className="flex items-center gap-3">
          <span className="w-8 h-px" style={{ background: 'var(--brass-dim)' }} />
          <span
            className="text-[11px] uppercase tracking-[0.28em] font-medium"
            style={{ color: 'var(--brass)' }}
          >
            Board Recognition
          </span>
          <span className="w-8 h-px" style={{ background: 'var(--brass-dim)' }} />
        </div>

        {/* Title */}
        <div className="text-center -mt-4">
          <h1
            className="text-engrave font-display text-6xl font-semibold mb-3 tracking-tight"
            style={{ color: 'var(--ivory)' }}
          >
            Let It Chessify
          </h1>
          <p className="text-base font-light" style={{ color: 'var(--stone)' }}>
            Photograph a board. Get the position. Analyse it.
          </p>
        </div>

        {/* Board type toggle */}
        <div
          className="flex items-center gap-1.5 rounded-full p-1.5 border"
          style={{ background: 'var(--walnut)', borderColor: 'var(--hairline)', boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.4)' }}
        >
          <button
            onClick={() => setIsPhysical(false)}
            className="text-sm font-medium px-5 py-2 rounded-full transition-all duration-200 flex items-center gap-1.5"
            style={
              !isPhysical
                ? { background: 'var(--felt)', color: 'var(--ivory)', boxShadow: '0 2px 8px rgba(78,122,91,0.35)' }
                : { color: 'var(--stone)' }
            }
          >
            <MonitorSmartphone size={13} />
            Screenshot
          </button>
          <button
            onClick={() => setIsPhysical(true)}
            className="text-sm font-medium px-5 py-2 rounded-full transition-all duration-200 flex items-center gap-1.5"
            style={
              isPhysical
                ? { background: 'var(--brass)', color: 'var(--walnut-deep)', boxShadow: '0 2px 8px rgba(201,161,92,0.35)' }
                : { color: 'var(--stone)' }
            }
          >
            <Camera size={13} />
            Physical board
          </button>
        </div>

        {/* Drop zone, framed like a board mid-detection */}
        <label
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`dropframe ${isDragging ? 'is-dragging' : ''} relative w-full max-w-lg aspect-video rounded-2xl
                      flex flex-col items-center justify-center gap-4 cursor-pointer
                      transition-all duration-300
                      ${loading ? 'opacity-60 pointer-events-none' : ''}`}
          style={{
            background: isDragging ? 'rgba(78,122,91,0.10)' : 'var(--walnut-raised)',
            border: `1px solid ${isDragging ? 'var(--felt-bright)' : 'var(--hairline)'}`,
            boxShadow: isDragging
              ? '0 0 0 1px rgba(111,163,120,0.25), 0 12px 40px rgba(0,0,0,0.45)'
              : '0 12px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03)',
          }}
        >
          {/* Corner brackets — a nod to the corner-detection step that follows */}
          <span className="corner-mark border-t-2 border-l-2 rounded-tl-lg" style={{ top: -1, left: -1 }} />
          <span className="corner-mark border-t-2 border-r-2 rounded-tr-lg" style={{ top: -1, right: -1 }} />
          <span className="corner-mark border-b-2 border-l-2 rounded-bl-lg" style={{ bottom: -1, left: -1 }} />
          <span className="corner-mark border-b-2 border-r-2 rounded-br-lg" style={{ bottom: -1, right: -1 }} />

          <input
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={onFileInput}
            disabled={loading}
          />

          {loading ? (
            <>
              <div
                className="w-10 h-10 border-2 border-t-transparent rounded-full animate-spin"
                style={{ borderColor: 'var(--felt-bright)', borderTopColor: 'transparent' }}
              />
              <p className="text-sm" style={{ color: 'var(--stone)' }}>Detecting board…</p>
            </>
          ) : (
            <>
              <div
                className="p-4 rounded-xl transition-colors duration-300"
                style={{ background: isDragging ? 'rgba(111,163,120,0.16)' : 'var(--walnut)' }}
              >
                <Upload size={26} style={{ color: isDragging ? 'var(--felt-bright)' : 'var(--brass)' }} />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium" style={{ color: 'var(--ivory)' }}>
                  Drop an image here
                </p>
                <p className="text-xs mt-1" style={{ color: 'var(--stone)' }}>
                  or click to browse — JPG, PNG, WebP
                </p>
              </div>
            </>
          )}
        </label>

        {/* Mode hint */}
        <p className="text-xs text-center max-w-sm" style={{ color: 'var(--stone)' }}>
          {isPhysical
            ? 'For best results, photograph from directly above with even lighting and all four corners visible.'
            : 'Works with Lichess, Chess.com, and any standard board screenshot.'}
        </p>

        {/* No photo? Skip straight to analysis from the standard setup, or
            skip the photo AND the standard position by importing a PGN.
            Mobile-only fallback -- on larger screens these live in the
            side strip instead. */}
        <div className="flex flex-col items-center gap-3">
          <div className="sm:hidden flex items-center gap-5 flex-wrap justify-center">
            {onStartFromScratch && (
              <button
                onClick={onStartFromScratch}
                disabled={loading}
                className="flex items-center gap-2 text-sm font-medium transition-colors
                           disabled:opacity-50 disabled:pointer-events-none"
                style={{ color: 'var(--brass)' }}
              >
                <Swords size={14} />
                No photo? Analyse from the starting position
              </button>
            )}
            {onImportPgn && (
              <button
                onClick={() => setShowPgnPanel((v) => !v)}
                disabled={loading}
                className="flex items-center gap-2 text-sm font-medium transition-colors
                           disabled:opacity-50 disabled:pointer-events-none"
                style={{ color: 'var(--brass)' }}
              >
                <FileUp size={14} />
                Import PGN
              </button>
            )}
          </div>

          {showPgnPanel && onImportPgn && (
            <div
              ref={pgnPanelRef}
              className="w-full max-w-lg rounded-xl p-4 flex flex-col gap-3"
              style={{ background: 'var(--walnut-raised)', border: '1px solid var(--hairline)' }}
            >
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium" style={{ color: 'var(--stone)' }}>
                  Paste PGN text, or load a .pgn file
                </p>
                <button
                  onClick={() => setShowPgnPanel(false)}
                  className="text-[#8A8A8A] hover:text-[#F5F0E8] transition-colors"
                  aria-label="Close"
                >
                  <X size={14} />
                </button>
              </div>

              <textarea
                ref={pgnTextareaRef}
                value={pgnText}
                onChange={(e) => setPgnText(e.target.value)}
                placeholder={'1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...'}
                rows={5}
                className="w-full rounded-lg px-3 py-2 text-xs font-mono resize-none
                           focus:outline-none"
                style={{
                  background: 'var(--walnut)',
                  border: '1px solid var(--hairline)',
                  color: 'var(--ivory)',
                }}
              />

              <div className="flex items-center gap-2">
                <button
                  onClick={submitPgn}
                  disabled={!pgnText.trim()}
                  className="flex-1 flex items-center justify-center gap-1.5 text-sm font-medium
                             px-4 py-2 rounded-lg transition-all disabled:opacity-40
                             disabled:pointer-events-none"
                  style={{ background: 'var(--felt)', color: 'var(--ivory)' }}
                >
                  <Swords size={13} />
                  Analyse this game
                </button>
                <button
                  onClick={() => pgnFileInputRef.current?.click()}
                  className="flex items-center justify-center gap-1.5 text-sm font-medium
                             px-4 py-2 rounded-lg transition-all"
                  style={{ border: '1px solid var(--hairline)', color: 'var(--stone)' }}
                >
                  <FolderOpen size={13} />
                  Choose file
                </button>
                <input
                  ref={pgnFileInputRef}
                  type="file"
                  accept=".pgn,text/plain"
                  className="sr-only"
                  onChange={onPgnFileInput}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
