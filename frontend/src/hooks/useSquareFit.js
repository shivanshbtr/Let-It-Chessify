import { useState, useRef, useEffect } from 'react'

// Chessboard v5 fills 100% width/height of its parent independently on each
// axis (there's no boardWidth prop), so it needs an explicitly *square* box
// to render into or it'll stretch/overflow. This measures the available
// space and hands back the largest square (in px) that fits, updating on
// resize.
export function useSquareFit() {
  const ref = useRef(null)
  const [size, setSize] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setSize(Math.max(0, Math.min(el.clientWidth, el.clientHeight)))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size]
}
