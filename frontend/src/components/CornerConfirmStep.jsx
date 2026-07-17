import { useState, useRef, useCallback, useEffect } from 'react'
import { Check, RotateCcw, ArrowLeft } from 'lucide-react'
import { refineCorners } from '../api/chess'

const HANDLE_RADIUS = 13
const LABELS = ['TL', 'TR', 'BR', 'BL']

// Compute the actual rendered rect of an `object-contain` image inside its box.
function getRenderRect(boxW, boxH, natW, natH) {
  if (!boxW || !boxH || !natW || !natH) {
    return { w: 0, h: 0, offsetX: 0, offsetY: 0 }
  }
  const boxRatio = boxW / boxH
  const imgRatio = natW / natH
  let w, h
  if (imgRatio > boxRatio) {
    // image is wider than box -> letterboxed top/bottom
    w = boxW
    h = boxW / imgRatio
  } else {
    // image is taller than box -> letterboxed left/right
    h = boxH
    w = boxH * imgRatio
  }
  return { w, h, offsetX: (boxW - w) / 2, offsetY: (boxH - h) / 2 }
}

export default function CornerConfirmStep({
  overlayB64,
  originalB64,
  initialCorners,
  initialGrid,
  isPhysical,
  onConfirm,
  onBack,
  loading,
}) {
  const [corners, setCorners]     = useState(initialCorners)
  const [grid, setGrid]           = useState(initialGrid)
  const [overlayUrl, setOverlayUrl] = useState(`data:image/png;base64,${overlayB64}`)
  const [dragging, setDragging]   = useState(null)
  const [refining, setRefining]   = useState(false)
  const [imgSize, setImgSize]     = useState({ w: 1, h: 1 })
  const [boxSize, setBoxSize]     = useState({ w: 0, h: 0 })
  const containerRef = useRef(null)
  const debounce = useRef(null)

  // Track natural (pixel) size of the source image
  useEffect(() => {
    const img = new Image()
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = `data:image/png;base64,${originalB64}`
  }, [originalB64])

  // Track the container box size (robust to layout/resize changes)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () => setBoxSize({ w: el.clientWidth, h: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const renderRect = getRenderRect(boxSize.w, boxSize.h, imgSize.w, imgSize.h)

  // Scale corner from image (pixel) coords -> on-screen coords within the container
  const toDisplay = ([x, y]) => ({
    x: renderRect.offsetX + (x / imgSize.w) * renderRect.w,
    y: renderRect.offsetY + (y / imgSize.h) * renderRect.h,
  })

  // Scale from on-screen (container-relative) coords -> image (pixel) coords, clamped to the photo
  const toImage = (dx, dy) => {
    if (!renderRect.w || !renderRect.h) return [0, 0]
    const cx = Math.max(renderRect.offsetX, Math.min(renderRect.offsetX + renderRect.w, dx))
    const cy = Math.max(renderRect.offsetY, Math.min(renderRect.offsetY + renderRect.h, dy))
    return [
      ((cx - renderRect.offsetX) / renderRect.w) * imgSize.w,
      ((cy - renderRect.offsetY) / renderRect.h) * imgSize.h,
    ]
  }

  const callRefine = useCallback(async (newCorners) => {
    setRefining(true)
    try {
      const res = await refineCorners(originalB64, newCorners, isPhysical)
      if (res.success) {
        setGrid(res.grid)
        setCorners(res.corners)
        setOverlayUrl(`data:image/png;base64,${res.overlay_image_b64}`)
      }
    } catch (e) {
      console.error('Refine failed:', e)
    }
    setRefining(false)
  }, [originalB64, isPhysical])

  const moveCorner = useCallback((idx, clientX, clientY) => {
    const rect = containerRef.current.getBoundingClientRect()
    const dx = clientX - rect.left
    const dy = clientY - rect.top
    const imgCoord = toImage(dx, dy)
    setCorners(prev => {
      const newCorners = prev.map((c, i) => i === idx ? imgCoord : c)
      clearTimeout(debounce.current)
      debounce.current = setTimeout(() => callRefine(newCorners), 150)
      return newCorners
    })
  }, [renderRect.w, renderRect.h, renderRect.offsetX, renderRect.offsetY, imgSize.w, imgSize.h, callRefine])

  // Each handle captures its own pointer, so corners can never interfere with one another
  const onHandlePointerDown = (e, idx) => {
    e.preventDefault()
    e.stopPropagation()
    e.currentTarget.setPointerCapture(e.pointerId)
    setDragging(idx)
  }

  const onHandlePointerMove = (e, idx) => {
    if (dragging !== idx) return
    e.preventDefault()
    moveCorner(idx, e.clientX, e.clientY)
  }

  const onHandlePointerUp = (e, idx) => {
    if (dragging !== idx) return
    try { e.currentTarget.releasePointerCapture(e.pointerId) } catch {}
    setDragging(null)
  }

  return (
    <div className="flex flex-col h-full gap-4 px-4 py-6">
      <div>
        <div className="flex items-center justify-between">
          <h2 className="font-display text-2xl font-semibold text-[#F5F0E8]">
            Confirm board boundary
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
          Drag any corner handle to adjust the grid, then confirm.
        </p>
      </div>

      {/* Image + overlay + corner handles */}
      <div
        ref={containerRef}
        className="relative flex-1 min-h-0 select-none"
      >
        {/* Original image */}
        <img
          src={`data:image/png;base64,${originalB64}`}
          alt="Board"
          className="w-full h-full object-contain rounded-xl"
          draggable={false}
        />

        {/* Grid overlay */}
        <img
          src={overlayUrl}
          alt="Grid overlay"
          className={`absolute inset-0 w-full h-full object-contain rounded-xl pointer-events-none transition-opacity ${
            refining ? 'opacity-50' : 'opacity-100'
          }`}
          draggable={false}
        />

        {/* Corner handles - positioned using the actual rendered photo rect, not the letterboxed box */}
        {renderRect.w > 0 && corners.map((corner, idx) => {
          const { x, y } = toDisplay(corner)
          return (
            <div
              key={idx}
              className="corner-handle absolute z-10 flex items-center justify-center"
              style={{
                left: x - HANDLE_RADIUS,
                top:  y - HANDLE_RADIUS,
                width:  HANDLE_RADIUS * 2,
                height: HANDLE_RADIUS * 2,
                touchAction: 'none',
                cursor: dragging === idx ? 'grabbing' : 'grab',
              }}
              onPointerDown={(e) => onHandlePointerDown(e, idx)}
              onPointerMove={(e) => onHandlePointerMove(e, idx)}
              onPointerUp={(e) => onHandlePointerUp(e, idx)}
              onPointerCancel={(e) => onHandlePointerUp(e, idx)}
            >
              <div
                className="w-5 h-5 rounded-full border-2 border-white shadow-lg pointer-events-none"
                style={{
                  background: '#6B9E6B',
                  transform: dragging === idx ? 'scale(1.3)' : 'scale(1)',
                  transition: 'transform 0.1s',
                }}
              />
              <span className="absolute -bottom-5 text-[10px] text-white font-medium pointer-events-none">
                {LABELS[idx]}
              </span>
            </div>
          )
        })}

        {refining && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-black/60 text-white text-xs px-3 py-1.5 rounded-full">
              Recalculating grid…
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => {
            setCorners(initialCorners)
            setGrid(initialGrid)
            setOverlayUrl(`data:image/png;base64,${overlayB64}`)
          }}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-[#333]
                     text-[#8A8A8A] hover:text-[#F5F0E8] hover:border-[#555] transition-all text-sm"
        >
          <RotateCcw size={14} />
          Reset
        </button>

        <button
          onClick={() => onConfirm(grid, corners)}
          disabled={loading}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                     bg-[#6B9E6B] hover:bg-[#7aaf7a] text-white font-medium text-sm
                     transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Classifying…
            </>
          ) : (
            <>
              <Check size={16} />
              Confirm grid
            </>
          )}
        </button>
      </div>
    </div>
  )
}
