import { create } from 'zustand'
import { AudioTransport } from './transport.ts'
import type { PlaybackSnapshot } from './transport.ts'
import type { Chart, Song } from './chart.ts'
import { downloadSong, importSongFile } from './songs.ts'
import type { SongResult } from './songs.ts'
import { loadCatalog, loadServerSong, withLiveCharts } from './catalog.ts'
import { sessionUrl } from './backend.ts'
import { SessionClient } from './session-client.ts'
import { useSession } from './session.ts'

type InspectorTab = 'activity' | 'songs' | 'session' | 'details' | 'next'

interface PlayerState extends PlaybackSnapshot {
  song: Song | null
  chart: Chart | null
  coverUrl: string | null
  loading: string
  error: string
  inspectorTab: InspectorTab
  brain: string
  // Inline type imports keep the shared import block untouched for parallel branches.
  brains: import('./catalog.ts').BrainOption[]
  serverMaps: import('./catalog.ts').ServerMap[]
  catalog: 'idle' | 'loading' | 'ready' | 'error'
  refreshCatalog: () => Promise<void>
  setBrain: (id: string) => void
  selectServer: (mapId: string) => Promise<void>
  setInspectorTab: (tab: InspectorTab) => void
  selectRemote: (result: SongResult, difficulty?: string, characteristic?: string) => Promise<void>
  importFile: (file: File) => Promise<void>
  selectChart: (id: string) => void
  cancelLoad: () => void
  clearError: () => void
  play: () => Promise<void>
  pause: () => void
  seek: (time: number) => void
  restart: () => void
  setRate: (rate: number) => void
  setVolume: (volume: number) => void
}

/** The server map a brain plays for the loaded chart, or undefined for a preview. */
export function liveMapId(s: { song: Song | null; chart: Chart | null }): string | undefined {
  return s.song?.serverMapId ?? (s.chart?.live ? s.chart.map_id : undefined)
}

let pending: AbortController | null = null
let playAttempt = 0
export const transport = new AudioTransport((snapshot) => {
  usePlayer.setState(snapshot)
  if (!snapshot.playing) liveSession.pause()
})
export const liveSession = new SessionClient({
  url: sessionUrl,
  getTime: () => transport.getTime(),
  onChange: (state) => useSession.setState(state),
  onFailure: () => {
    transport.pause()
  },
})

function cancelSession() {
  playAttempt++
  liveSession.disconnect()
}

async function loadSong(
  label: string,
  loader: (signal: AbortSignal) => Promise<Song>,
  difficulty?: string,
  characteristic?: string,
) {
  pending?.abort()
  const controller = new AbortController()
  pending = controller
  cancelSession()
  transport.pause()
  usePlayer.setState({ loading: label, error: '' })
  try {
    const song = await loader(controller.signal)
    controller.signal.throwIfAborted()
    usePlayer.setState({ loading: `Preparing audio for ${song.title}…` })
    const audio = await transport.decode(song.audio)
    controller.signal.throwIfAborted()
    const oldCover = usePlayer.getState().coverUrl
    const chart =
      song.charts.find(
        (c) =>
          (c.map_id === difficulty || c.difficulty.name === difficulty) &&
          (!characteristic || c.characteristic === characteristic),
      ) ??
      song.charts.find((c) => c.characteristic === 'Standard' && c.difficulty.name === 'Hard') ??
      song.charts[0]
    const coverUrl = song.cover ? URL.createObjectURL(song.cover) : null
    transport.load(audio)
    if (liveMapId({ song, chart })) transport.setRate(1)
    usePlayer.setState({ song, chart, coverUrl, loading: '', error: '' })
    if (oldCover) URL.revokeObjectURL(oldCover)
  } catch (cause) {
    if (!controller.signal.aborted)
      usePlayer.setState({
        loading: '',
        error:
          cause instanceof Error && cause.name === 'TimeoutError'
            ? 'The song download timed out. Try again.'
            : cause instanceof TypeError
              ? 'Could not download this song. Check your connection and try again.'
              : cause instanceof Error
                ? cause.message
                : 'Could not load this song. Try again or choose another map.',
      })
  } finally {
    if (pending === controller) pending = null
  }
}

export const usePlayer = create<PlayerState>((set, get) => ({
  song: null,
  chart: null,
  coverUrl: null,
  loading: '',
  error: '',
  inspectorTab: 'activity',
  brain: '',
  brains: [],
  serverMaps: [],
  catalog: 'idle',
  refreshCatalog: async () => {
    if (get().catalog === 'loading') return
    set({ catalog: 'loading' })
    try {
      const { maps, brains } = await loadCatalog(new AbortController().signal)
      // Without a brain, live songs refuse to start and the fly never moves.
      const brain = brains.some((b) => b.id === get().brain)
        ? get().brain
        : ((brains.find((b) => b.id === 'real') ?? brains[0])?.id ?? '')
      set({ brains, serverMaps: maps, brain, catalog: 'ready' })
    } catch {
      set({ catalog: 'error' })
    }
  },
  setBrain: (brain) => {
    cancelSession()
    transport.pause()
    set({ brain })
  },
  selectServer: (mapId) =>
    loadSong('Loading server song…', (signal) => loadServerSong(mapId, signal)),
  time: 0,
  duration: 0,
  playing: false,
  rate: 1,
  volume: 0.7,
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
  selectRemote: (result, difficulty, characteristic) =>
    loadSong(
      `Downloading ${result.title}…`,
      async (signal) => withLiveCharts(await downloadSong(result, signal), result.id, signal),
      difficulty,
      characteristic,
    ),
  importFile: (file) => loadSong(`Opening ${file.name}…`, (signal) => importSongFile(file, signal)),
  selectChart: (id) => {
    if (get().loading) return
    const chart = get().song?.charts.find((c) => c.map_id === id)
    if (chart) {
      cancelSession()
      if (chart.live) transport.setRate(1)
      set({ chart })
    }
  },
  cancelLoad: () => {
    pending?.abort()
    pending = null
    set({ loading: '' })
  },
  clearError: () => set({ error: '' }),
  play: async () => {
    if (get().loading || !get().chart || useSession.getState().status === 'connecting') return
    const attempt = ++playAttempt
    try {
      const mapId = liveMapId(get())
      if (mapId) {
        if (!get().brain)
          throw new Error('Choose an available brain in the Songs tab, then press play.')
        await transport.unlock()
        if (attempt !== playAttempt) return
        if (get().duration > 0 && transport.getTime() >= get().duration) liveSession.disconnect()
        await liveSession.prepare(mapId, get().brain)
        if (attempt !== playAttempt) return
      }
      await transport.play()
      if (attempt !== playAttempt) {
        transport.pause()
        return
      }
      if (transport.snapshot().playing) liveSession.play()
      set({ error: '' })
    } catch (cause) {
      if (attempt !== playAttempt) return
      transport.pause()
      set({
        error:
          liveMapId(get()) && cause instanceof Error
            ? cause.message
            : 'Audio could not start. Press play again, or check your browser’s audio permissions.',
      })
    }
  },
  pause: () => {
    playAttempt++
    if (useSession.getState().status === 'connecting') liveSession.disconnect()
    liveSession.pause()
    transport.pause()
  },
  seek: (time) => {
    if (!get().loading && Number.isFinite(time)) {
      transport.seek(time)
      liveSession.seek()
    }
  },
  restart: () => {
    if (!get().loading) {
      get().pause()
      transport.seek(0)
      liveSession.seek()
    }
  },
  setRate: (rate) => {
    if (!liveMapId(get())) transport.setRate(rate)
  },
  setVolume: (volume) => transport.setVolume(volume),
}))

export function openSongLibrary() {
  usePlayer.getState().setInspectorTab('songs')
  if (window.innerWidth < 900)
    requestAnimationFrame(() => {
      document.querySelector('.inspector')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
}
