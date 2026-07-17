const STEPS = [
  { id: 1, label: 'Upload' },
  { id: 2, label: 'Grid' },
  { id: 3, label: 'Editor' },
  { id: 4, label: 'Turn' },
  { id: 5, label: 'Analysis' },
]

// availableSteps: ids the user can actually jump to right now (they either
// already have the data for that step, or it's the step being shown).
// onStepClick: called with a step id when an available, non-active dot is clicked.
export default function StepIndicator({ currentStep, availableSteps = [], onStepClick }) {
  return (
    <div className="flex flex-col gap-1 py-8 px-3 w-20 flex-shrink-0">
      {/* Logo */}
      <div className="mb-8 text-center">
        <span className="font-display text-[#C8A96E] text-lg font-semibold">♛</span>
      </div>

      {STEPS.map((step, i) => {
        const available = availableSteps.includes(step.id)
        const active    = currentStep === step.id
        const done      = available && step.id < currentStep
        const clickable = available && !active && !!onStepClick

        return (
          <div key={step.id} className="flex flex-col items-center">
            <div className="flex flex-col items-center gap-1">
              <button
                type="button"
                onClick={() => clickable && onStepClick(step.id)}
                disabled={!clickable}
                title={clickable ? `Go to ${step.label}` : undefined}
                className={`step-dot ${active ? 'active' : done ? 'done' : ''} ${
                  clickable ? 'cursor-pointer hover:scale-125' : 'cursor-default'
                }`}
                style={{ transition: 'transform 0.15s' }}
              />
              <span className={`text-[9px] uppercase tracking-wider text-center leading-tight ${
                active ? 'text-[#6B9E6B]' : done ? 'text-[#C8A96E]' : 'text-[#333]'
              }`}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`w-px h-6 mt-1 transition-colors ${
                done ? 'bg-[#C8A96E]' : 'bg-[#2a2a2a]'
              }`} />
            )}
          </div>
        )
      })}
    </div>
  )
}
