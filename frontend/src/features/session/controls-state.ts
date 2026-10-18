import { create } from 'zustand'
import { isUntouched, noControls } from '@/features/session/interventions'
import type { BrainControls } from '@/features/session/interventions'
import { liveSession } from '@/features/session/store'

// Control groups and moods named in controls.json.
export const LOOMING = 'looming_detectors'
export const AROUSAL = 'arousal'

/** Brain-control state for the player, shared by the eyes, the Brain stage and Session. */
export const useBrainControls = create<{
  controls: BrainControls
  change: (next: Partial<BrainControls>) => void
}>((set, get) => ({
  controls: noControls,
  change: (next) => {
    const controls = { ...get().controls, ...next }
    set({ controls })
    liveSession.setInterventions(controls)
  },
}))

/** Plain-language summary of the active interventions, or null when untouched. */
export function describeControls(controls: BrainControls): string | null {
  if (isUntouched(controls)) return null
  const parts: string[] = []
  if (controls.lesion.includes(LOOMING)) parts.push('looming detectors lesioned')
  if (controls.blindLeft && controls.blindRight) parts.push('both eyes blindfolded')
  else if (controls.blindLeft) parts.push('left eye blindfolded')
  else if (controls.blindRight) parts.push('right eye blindfolded')
  const arousal = controls.mood[AROUSAL] ?? 0
  if (arousal) parts.push(`arousal ${arousal > 0 ? '+' : ''}${arousal.toFixed(2)}`)
  const text = parts.join(', ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}
