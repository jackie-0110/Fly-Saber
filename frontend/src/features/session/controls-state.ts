import { useEffect } from 'react'
import { create } from 'zustand'
import { loadControls } from '@/features/brain/data'
import type { Controls } from '@/features/brain/data'
import { isUntouched, noControls } from '@/features/session/interventions'
import type { BrainControls } from '@/features/session/interventions'
import { liveSession } from '@/features/session/store'

// Control groups and moods named in controls.json.
export const LOOMING = 'looming_detectors'
export const AROUSAL = 'arousal'
export const GIANT_FIBER = 'giant_fiber'

/** Brain-control state for the player, shared by the eyes, the Brain stage, the panel and Session. */
export const useBrainControls = create<{
  controls: BrainControls
  change: (next: Partial<BrainControls>) => void
  /** Whether the full controls panel is open over the inspector. */
  panelOpen: boolean
  setPanelOpen: (open: boolean) => void
}>((set, get) => ({
  controls: noControls,
  panelOpen: false,
  setPanelOpen: (panelOpen) => set({ panelOpen }),
  change: (next) => {
    const controls = { ...get().controls, ...next }
    set({ controls })
    liveSession.setInterventions(controls)
  },
}))

// controls.json, loaded once and shared by everything that shows a control.
const useControlsConfig = create<{ config: Controls | null; failed: boolean }>(() => ({
  config: null,
  failed: false,
}))
let configRequest: Promise<void> | null = null
function requestConfig() {
  configRequest ??= loadControls(new AbortController().signal)
    .then((config) => useControlsConfig.setState({ config, failed: false }))
    .catch(() => {
      configRequest = null
      useControlsConfig.setState({ failed: true })
    })
}

/** controls.json once loaded (null until then), and whether loading failed. */
export function useControlsConfigState() {
  const state = useControlsConfig()
  useEffect(requestConfig, [])
  return state
}

const sentence = (parts: string[]) => {
  const text = parts.join(', ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}
const signed = (level: number) => `${level > 0 ? '+' : ''}${level.toFixed(2)}`

/**
 * Plain-language summary of the active interventions, or null when untouched. Labels come
 * from controls.json when it has loaded, else from the group and mood names.
 */
export function describeControls(controls: BrainControls, config?: Controls | null) {
  if (isUntouched(controls)) return null
  const section = (group: string) =>
    (config?.sections.find((s) => s.group === group)?.label ?? group.replaceAll('_', ' '))
      .replace(/ \(.*\)$/, '')
      .toLowerCase()
  const mood = (name: string) =>
    (config?.moods.find((m) => m.name === name)?.label ?? name.replaceAll('_', ' ')).toLowerCase()
  const parts: string[] = []
  if (controls.lesion.length) parts.push(`${controls.lesion.map(section).join(' and ')} lesioned`)
  for (const [group, current] of Object.entries(controls.stimulate))
    if (current) parts.push(`${section(group)} stimulated at ${current.toFixed(1)}`)
  for (const [name, level] of Object.entries(controls.mood))
    if (level) parts.push(`${mood(name)} ${signed(level)}`)
  if (controls.noise) parts.push('neural noise on')
  if (controls.blindLeft && controls.blindRight) parts.push('both eyes blindfolded')
  else if (controls.blindLeft) parts.push('left eye blindfolded')
  else if (controls.blindRight) parts.push('right eye blindfolded')
  if (controls.blurDeg) parts.push(`vision blurred ${controls.blurDeg.toFixed(1)}°`)
  if (controls.facetDropout)
    parts.push(`${Math.round(controls.facetDropout * 100)}% of facets dropped`)
  if (controls.eyeNoise) parts.push(`visual noise ${controls.eyeNoise.toFixed(3)}`)
  return sentence(parts)
}
