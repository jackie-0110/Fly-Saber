import { create } from 'zustand'
import { AudioTransport } from './transport.ts'
import type { PlaybackSnapshot } from './transport.ts'
import type { Chart, Song } from './chart.ts'
import { downloadSong, importSongFile } from './songs.ts'
import type { SongResult } from './songs.ts'
import { loadServerSong } from './catalog.ts'
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
    if (song.serverMapId) transport.setRate(1)
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
      (signal) => downloadSong(result, signal),
      difficulty,
      characteristic,
    ),
  importFile: (file) => loadSong(`Opening ${file.name}…`, (signal) => importSongFile(file, signal)),
  selectChart: (id) => {
    if (get().loading) return
    const chart = get().song?.charts.find((c) => c.map_id === id)
    if (chart) {
      cancelSession()
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
      if (get().song?.serverMapId) {
        if (!get().brain)
          throw new Error('Choose an available brain in the Songs tab, then press play.')
        await transport.unlock()
        if (attempt !== playAttempt) return
        if (get().duration > 0 && transport.getTime() >= get().duration) liveSession.disconnect()
        await liveSession.prepare(get().song!.serverMapId!, get().brain)
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
          get().song?.serverMapId && cause instanceof Error
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
    if (!get().song?.serverMapId) transport.setRate(rate)
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
