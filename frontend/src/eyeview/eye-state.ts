import { create } from 'zustand'
import { backendJson } from '../game/backend.ts'
import { parseEyeLattice } from './eye-data.ts'
import type { EyeFrame, EyeLattice } from './eye-data.ts'

interface EyeState {
  frames: Record<number, EyeFrame | null>
  lattice: EyeLattice | null
  brain: string
  error: boolean
  expanded: boolean
}

export const useEyeView = create<EyeState>(() => ({
  frames: {},
  lattice: null,
  brain: '',
  error: false,
  expanded: false,
}))

let request = 0
export async function loadEyeLattice(brain: string, signal: AbortSignal) {
  const id = ++request
  useEyeView.setState({ lattice: null, brain, error: false })
  if (!brain) return
  try {
    const lattice = parseEyeLattice(
      await backendJson(`/api/eye_lattice?brain=${encodeURIComponent(brain)}`, signal),
    )
    if (id === request && !signal.aborted) useEyeView.setState({ lattice })
  } catch {
    if (id === request && !signal.aborted) useEyeView.setState({ error: true })
  }
}

export const liveEye = {
  receive(frame: EyeFrame) {
    useEyeView.setState((state) => ({ frames: { ...state.frames, [frame.slot]: frame } }))
  },
  clear() {
    useEyeView.setState({ frames: {} })
  },
}
