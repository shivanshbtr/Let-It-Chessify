// Vertical score bar: white fills from center upward, black from center downward
// evalCp: centipawn score from white's perspective
// mateIn: null or integer (positive = white mating, negative = black mating;
//         can also legitimately be 0 for an already-checkmated position)

function cpToPercent(cp) {
  if (cp === null || cp === undefined) return 50
  // Sigmoid-like: 50% at 0, approaches 0/100 at ±1000cp
  const clamped = Math.max(-1000, Math.min(1000, cp))
  return 50 + 50 * (clamped / 1000) * (2 - Math.abs(clamped) / 1000)
}

export default function ScoreBar({ evalCp, evalType, mateIn }) {
  const isMate = evalType === 'mate'

  let whitePct
  if (isMate) {
    // Use evalCp's sign, not mateIn's. evalCp is always a clean ±10000 for
    // a mate eval and is never itself 0 -- whereas mateIn CAN legitimately
    // be 0 (an already-checkmated position), and plain 0 has no sign to
    // compare against, which is what made the bar always default to
    // "black winning" on a white checkmate.
    whitePct = evalCp > 0 ? 98 : 2
  } else {
    whitePct = cpToPercent(evalCp)
  }

  const blackPct = 100 - whitePct
  // Keep the label fully inside the track even at extreme (near-98/2) splits.
  const labelTop = Math.min(92, Math.max(8, blackPct))

  let label
  if (isMate) {
    label = `M${Math.abs(mateIn)}`
  } else if (evalCp === null || evalCp === undefined) {
    label = '0.0'
  } else {
    const absVal = Math.abs(evalCp) / 100
    const digits = absVal >= 10 ? absVal.toFixed(0) : absVal.toFixed(1)
    label = evalCp > 0 ? `+${digits}` : evalCp < 0 ? `-${digits}` : digits
  }

  return (
    <div className="flex flex-col items-center h-full w-10 select-none">
      <div className="relative flex-1 w-full rounded-full overflow-hidden border border-[#333] shadow-[inset_0_1px_3px_rgba(0,0,0,0.5)]">
        {/* Black section (top) -- deliberately darker than the page/container
            background (#1A1A1A) so it actually reads as a filled section
            instead of blending invisibly into the surrounding UI. */}
        <div
          className="score-fill absolute top-0 left-0 right-0 bg-[#0A0A0A]"
          style={{ height: `${blackPct}%` }}
        />
        {/* White section (bottom) */}
        <div
          className="score-fill absolute bottom-0 left-0 right-0 bg-[#F5F0E8]"
          style={{ height: `${whitePct}%` }}
        />
        {/* Crisp divider right at the black/white boundary, so the split
            reads clearly at a glance instead of relying on color contrast alone. */}
        <div
          className="score-fill absolute left-0 right-0 h-[2px] bg-[#C8A96E]"
          style={{ top: `${blackPct}%`, transform: 'translateY(-50%)' }}
        />

        {/* Score label in its own pill -- always legible regardless of
            whether it lands over the black or white section. */}
        <div
          className="score-fill absolute left-1/2 px-1.5 py-0.5 rounded-full
                      bg-[#C8A96E] text-[#1A1A1A] text-[9px] font-bold
                      shadow-md whitespace-nowrap"
          style={{ top: `${labelTop}%`, transform: 'translate(-50%, -50%)' }}
        >
          {label}
        </div>
      </div>

      {/* W / B labels */}
      <div className="flex flex-col items-center gap-0.5 mt-1.5">
        <span className="text-[8px] text-[#F5F0E8] font-medium">W</span>
        <span className="text-[8px] text-[#8A8A8A]">B</span>
      </div>
    </div>
  )
}
