import { create } from 'zustand'
import { AudioTransport } from './transport.ts'
import type { PlaybackSnapshot } from './transport.ts'
import type { Chart, Song } from './chart.ts'
import { downloadSong, importSongFile } from './songs.ts'
import type { SongResult } from './songs.ts'

type InspectorTab = 'activity' | 'songs' | 'session' | 'details' | 'next'

interface PlayerState extends PlaybackSnapshot {
  song: Song | null
  chart: Chart | null
  coverUrl: string | null
  loading: string
  error: string
  inspectorTab: InspectorTab
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
export const transport = new AudioTransport((snapshot) => usePlayer.setState(snapshot))

async function loadSong(
  label: string,
  loader: (signal: AbortSignal) => Promise<Song>,
  difficulty?: string,
  characteristic?: string,
) {
  pending?.abort()
  const controller = new AbortController()
  pending = controller
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
    if (chart) set({ chart })
  },
  cancelLoad: () => {
    pending?.abort()
    pending = null
    set({ loading: '' })
  },
  clearError: () => set({ error: '' }),
  play: async () => {
    if (get().loading) return
    try {
      await transport.play()
      set({ error: '' })
    } catch {
      set({
        error:
          'Audio could not start. Press play again, or check your browser’s audio permissions.',
      })
    }
  },
  pause: () => transport.pause(),
  seek: (time) => {
    if (!get().loading) transport.seek(time)
  },
  restart: () => {
    if (!get().loading) {
      transport.pause()
      transport.seek(0)
    }
  },
  setRate: (rate) => transport.setRate(rate),
  setVolume: (volume) => transport.setVolume(volume),
}))

export function openSongLibrary() {
  usePlayer.getState().setInspectorTab('songs')
  if (window.innerWidth < 900)
    requestAnimationFrame(() => {
      document.querySelector('.inspector')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
}
