import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { loadCatalog } from '@/game/catalog'
import type { BrainOption, ServerMap } from '@/game/catalog'
import { difficultyLabel } from '@/game/chart'
import { usePlayer } from '@/game/store'

export function LiveSongs() {
  const [maps, setMaps] = useState<ServerMap[]>([])
  const [brains, setBrains] = useState<BrainOption[]>([])
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const brain = usePlayer((s) => s.brain)
  const loaded = usePlayer((s) => s.song?.serverMapId)
  const loading = usePlayer((s) => s.loading)
  useEffect(() => {
    const controller = new AbortController()
    loadCatalog(controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return
        setMaps(catalog.maps)
        setBrains(catalog.brains)
        if (!catalog.brains.some((b) => b.id === usePlayer.getState().brain))
          usePlayer
            .getState()
            .setBrain(
              catalog.brains.find((b) => b.id === 'real')?.id ?? catalog.brains[0]?.id ?? '',
            )
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error
              ? cause.message
              : 'Could not reach the server catalog. Retry when the backend is running.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false)
      })
    return () => controller.abort()
  }, [attempt])

  return (
    <div className="song-library live-library" aria-busy={busy}>
      <h2>Live songs</h2>
      <p className="library-intro">
        Choose a server song and a player. Press play below the track to start.
      </p>
      <label className="live-brain-picker">
        Player
        <select
          aria-label="Brain player"
          value={brain}
          disabled={busy || !brains.length || !!loading}
          onChange={(event) => usePlayer.getState().setBrain(event.target.value)}
        >
          {!brains.length && <option value="">No players available</option>}
          {brains.map((b) => (
            <option key={b.id} value={b.id}>
              {b.label}
            </option>
          ))}
        </select>
      </label>
      {brains.find((b) => b.id === brain) && (
        <p className="library-intro">{brains.find((b) => b.id === brain)!.description}</p>
      )}
      {busy && <p role="status">Loading server songs and players…</p>}
      {error && <p role="alert">{error}</p>}
      {!busy && !error && !maps.length && (
        <p>No server songs are available yet. Refresh after the demo maps are installed.</p>
      )}
      <div className="song-results">
        {maps.map((map) => (
          <article
            className={`song-card${loaded === map.map_id ? ' is-loaded' : ''}`}
            key={map.map_id}
          >
            <button
              className="song-card-main"
              type="button"
              disabled={!!loading || !brains.length}
              aria-label={`Load live ${map.song.title} ${difficultyLabel(map.difficulty.name)}`}
              onClick={() => void usePlayer.getState().selectServer(map.map_id)}
            >
              <div className="song-card-copy">
                <h3>{map.song.title}</h3>
                <p>{map.song.artist}</p>
                <small>Mapped by {map.song.mapper}</small>
              </div>
              <span className="current-difficulty" data-difficulty={map.difficulty.name}>
                {difficultyLabel(map.difficulty.name)}
              </span>
            </button>
            <div className="song-card-meta">
              <span>{map.note_count.toLocaleString()} notes</span>
              <span>{map.split === 'heldout' ? 'Unseen map' : 'Training map'}</span>
            </div>
            {map.song.license && <p className="catalog-license">{map.song.license}</p>}
          </article>
        ))}
      </div>
      <Button
        variant="outline"
        size="sm"
        disabled={busy}
        onClick={() => {
          setBusy(true)
          setError('')
          setAttempt((n) => n + 1)
        }}
      >
        Refresh server catalog
      </Button>
    </div>
  )
}
