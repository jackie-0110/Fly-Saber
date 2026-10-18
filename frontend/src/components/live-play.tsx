import { useEffect, useMemo } from 'react'
import { LoaderCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { difficultyLabel } from '@/game/chart'
import type { ServerMap } from '@/game/catalog'
import { usePlayer } from '@/game/store'

const DIFFICULTIES = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

/** Brain picker and the server's own maps: the songs a brain is guaranteed to play. */
export function LivePlay() {
  const brains = usePlayer((s) => s.brains)
  const brain = usePlayer((s) => s.brain)
  const maps = usePlayer((s) => s.serverMaps)
  const catalog = usePlayer((s) => s.catalog)
  const chart = usePlayer((s) => s.chart)
  const loading = usePlayer((s) => s.loading)

  useEffect(() => {
    const { refreshCatalog } = usePlayer.getState()
    void refreshCatalog()
    // The backend can come up after the page; keep trying until the catalog loads.
    const retry = setInterval(() => {
      if (usePlayer.getState().catalog === 'error') void refreshCatalog()
    }, 10000)
    return () => clearInterval(retry)
  }, [])

  const songs = useMemo(() => {
    const grouped = new Map<string, ServerMap[]>()
    for (const map of maps) {
      const key = `${map.song.title}\u0000${map.song.artist}`
      grouped.set(key, [...(grouped.get(key) ?? []), map])
    }
    const rank = (map: ServerMap) => DIFFICULTIES.indexOf(map.difficulty.name)
    return [...grouped.values()].map((group) => group.sort((a, b) => rank(a) - rank(b)))
  }, [maps])
  const selected = brains.find((b) => b.id === brain)

  return (
    <section className="live-play" aria-labelledby="live-play-heading">
      <div className="library-heading">
        <h2 id="live-play-heading">Live play</h2>
      </div>
      <p className="library-intro">
        The fly moves when a brain plays the song on the server. Choose a brain, then a song.
      </p>
      <label className="brain-picker">
        Fly brain
        <select
          aria-label="Fly brain"
          value={brain}
          disabled={!brains.length}
          onChange={(event) => usePlayer.getState().setBrain(event.target.value)}
        >
          {!brains.length && <option value="">No brains available</option>}
          {brains.map((option) => (
            <option key={option.id} value={option.id}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      {selected && <p className="brain-description">{selected.description}</p>}
      {catalog === 'error' && (
        <div className="library-message" role="status">
          <p>The backend is unreachable, so songs play as previews and the fly stays still.</p>
          <Button
            variant="outline"
            size="xs"
            onClick={() => void usePlayer.getState().refreshCatalog()}
          >
            Retry
          </Button>
        </div>
      )}
      {catalog === 'loading' && !maps.length && (
        <p className="library-loading" role="status">
          <LoaderCircle className="spin" />
          Loading server songs…
        </p>
      )}
      {songs.length > 0 && (
        <div className="song-results" aria-label="Server songs">
          {songs.map((group) => (
            <article className="song-card" key={group[0].map_id}>
              <div className="server-song-copy song-card-copy">
                <h3>{group[0].song.title}</h3>
                <p>{group[0].song.artist}</p>
                <small>Mapped by {group[0].song.mapper}</small>
              </div>
              <div className="difficulty-buttons">
                {group.map((map) => (
                  <button
                    type="button"
                    key={map.map_id}
                    data-difficulty={map.difficulty.name}
                    aria-pressed={chart?.map_id === map.map_id}
                    disabled={!!loading}
                    title={`${map.note_count} notes`}
                    onClick={() => void usePlayer.getState().selectServer(map.map_id)}
                  >
                    {difficultyLabel(map.difficulty.name)}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
