import { useEffect, useMemo } from 'react'
import { Check, LoaderCircle, Music2, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { difficultyLabel } from '@/game/chart'
import type { ServerMap } from '@/game/catalog'
import { usePlayer } from '@/game/store'

const DIFFICULTIES = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']

/** The brain that plays every live song. Loads the server catalog on mount. */
export function BrainPicker() {
  const brains = usePlayer((s) => s.brains)
  const brain = usePlayer((s) => s.brain)
  const catalog = usePlayer((s) => s.catalog)

  useEffect(() => {
    const { refreshCatalog } = usePlayer.getState()
    void refreshCatalog()
    // The backend can come up after the page; keep trying until the catalog loads.
    const retry = setInterval(() => {
      if (usePlayer.getState().catalog === 'error') void refreshCatalog()
    }, 10000)
    return () => clearInterval(retry)
  }, [])

  const selected = brains.find((b) => b.id === brain)
  return (
    <div className="brain-picker-block">
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
    </div>
  )
}

function matches(map: ServerMap, query: string) {
  const text = query.trim().toLowerCase()
  return (
    !text ||
    [map.song.title, map.song.artist, map.song.mapper].some((v) => v.toLowerCase().includes(text))
  )
}

/**
 * The server's own maps as ordinary song cards, pinned above community results and
 * filtered by the same search. These always go live, so they carry an "Always live" tag.
 */
export function ServerSongs({ query }: { query: string }) {
  const maps = usePlayer((s) => s.serverMaps)
  const catalog = usePlayer((s) => s.catalog)
  const chart = usePlayer((s) => s.chart)
  const loading = usePlayer((s) => s.loading)

  const songs = useMemo(() => {
    const grouped = new Map<string, ServerMap[]>()
    for (const map of maps.filter((m) => matches(m, query))) {
      const key = `${map.song.title}\u0000${map.song.artist}`
      grouped.set(key, [...(grouped.get(key) ?? []), map])
    }
    const rank = (map: ServerMap) => DIFFICULTIES.indexOf(map.difficulty.name)
    return [...grouped.values()].map((group) => group.sort((a, b) => rank(a) - rank(b)))
  }, [maps, query])

  if (catalog === 'loading' && !maps.length)
    return (
      <p className="library-loading" role="status">
        <LoaderCircle className="spin" />
        Loading ready-to-play songs…
      </p>
    )
  if (!songs.length) return null
  return (
    <>
      <div className="library-section-label">
        <span>Ready to play</span>
        <span>{songs.length === 1 ? '1 song' : `${songs.length} songs`}</span>
      </div>
      <div className="song-results" aria-label="Server songs">
        {songs.map((group) => {
          const loaded = group.some((map) => map.map_id === chart?.map_id)
          // Card click plays Expert when the song has it: the brains were trained there.
          const main = group.find((map) => map.difficulty.name === 'Expert') ?? group.at(-1)!
          return (
            <article className={`song-card${loaded ? ' is-loaded' : ''}`} key={group[0].map_id}>
              <button
                type="button"
                className="song-card-main"
                disabled={!!loading}
                onClick={() => void usePlayer.getState().selectServer(main.map_id)}
                aria-label={`Load ${group[0].song.title} by ${group[0].song.artist}`}
              >
                <div className="song-cover">
                  <Music2 aria-hidden="true" />
                </div>
                <div className="song-card-copy">
                  <h3>{group[0].song.title}</h3>
                  <p>{group[0].song.artist}</p>
                  <small>Mapped by {group[0].song.mapper}</small>
                </div>
                {loaded ? (
                  <Check className="song-load-icon" />
                ) : (
                  <Play className="song-load-icon" />
                )}
              </button>
              <div className="song-card-meta">
                <span className="live-tag">Always live</span>
                <span>{group[0].song.license ?? 'Server map'}</span>
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
          )
        })}
      </div>
    </>
  )
}
