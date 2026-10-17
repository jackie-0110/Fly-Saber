import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowDownToLine, ArrowUpRight, Check, LoaderCircle, Music2, Search, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { difficultyLabel, formatTime } from '@/game/chart'
import { searchSongs } from '@/game/songs'
import type { SongResult } from '@/game/songs'
import { usePlayer } from '@/game/store'
import { LiveSongs } from '@/components/live-songs'

export function SongLibrary() {
  const active = usePlayer((s) => s.inspectorTab === 'songs')
  const [source, setSource] = useState<'live' | 'community'>('live')
  return (
    <>
      <div className="library-sources" role="group" aria-label="Song source">
        <Button
          variant={source === 'live' ? 'secondary' : 'ghost'}
          size="sm"
          aria-pressed={source === 'live'}
          onClick={() => setSource('live')}
        >
          Live songs
        </Button>
        <Button
          variant={source === 'community' ? 'secondary' : 'ghost'}
          size="sm"
          aria-pressed={source === 'community'}
          onClick={() => setSource('community')}
        >
          Community preview
        </Button>
      </div>
      {source === 'live' ? active && <LiveSongs /> : <CommunitySongLibrary />}
    </>
  )
}

function CommunitySongLibrary() {
  const active = usePlayer((s) => s.inspectorTab === 'songs')
  const loaded = usePlayer((s) => s.song)
  const loading = usePlayer((s) => s.loading)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('Latest')
  const [results, setResults] = useState<SongResult[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const [more, setMore] = useState(false)
  const [searched, setSearched] = useState(false)
  const [lastQuery, setLastQuery] = useState({ query: '', sort: 'Latest' })
  const request = useRef<AbortController | null>(null)
  const initial = useRef(false)

  const search = useCallback(async (text: string, order: string, nextPage = 0) => {
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setBusy(true)
    setError('')
    if (nextPage === 0) {
      setResults([])
      setMore(false)
      setLastQuery({ query: text, sort: order })
    }
    try {
      const response = await searchSongs(text, nextPage, order, controller.signal)
      if (controller.signal.aborted) return
      setResults((previous) =>
        nextPage === 0
          ? response.songs
          : [
              ...previous,
              ...response.songs.filter((song) => !previous.some((s) => s.id === song.id)),
            ],
      )
      setPage(nextPage)
      setMore(response.more)
      setSearched(true)
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(
          cause instanceof Error && cause.name !== 'TypeError'
            ? cause.message
            : 'Could not reach song search. Try again in a moment.',
        )
    } finally {
      if (!controller.signal.aborted) setBusy(false)
    }
  }, [])

  useEffect(() => {
    if (active && !initial.current) {
      initial.current = true
      void search('', 'Latest')
    }
  }, [active, search])
  useEffect(
    () => () => {
      request.current?.abort()
      // Let StrictMode's effect replay restart the initial search.
      initial.current = false
    },
    [],
  )

  function choose(result: SongResult, name?: string, characteristic?: string) {
    if (loaded?.id === result.id) {
      const chart = loaded.charts.find(
        (c) =>
          (!name || c.difficulty.name === name) &&
          (!characteristic || c.characteristic === characteristic),
      )
      if (chart) usePlayer.getState().selectChart(chart.map_id)
      else void usePlayer.getState().selectRemote(result, name, characteristic)
    } else void usePlayer.getState().selectRemote(result, name, characteristic)
  }

  return (
    <div className="song-library">
      <div className="library-heading">
        <h2>Community maps</h2>
      </div>
      <p className="library-intro">
        Browse community maps. The server fetches the chart when you load one, so the fly can play
        its Standard difficulties.
      </p>
      <form
        className="song-search"
        onSubmit={(event) => {
          event.preventDefault()
          void search(query, sort)
        }}
      >
        <div className="song-search-input">
          <Search aria-hidden="true" />
          <input
            aria-label="Search songs"
            placeholder="Song, artist, or map link"
            value={query}
            maxLength={200}
            onChange={(event) => setQuery(event.target.value)}
          />
          {query && (
            <button type="button" onClick={() => setQuery('')} aria-label="Clear song search">
              <X />
            </button>
          )}
        </div>
        <div className="song-search-actions">
          <label>
            <span className="sr-only">Song sort order</span>
            <select
              aria-label="Song sort order"
              value={sort}
              onChange={(event) => setSort(event.target.value)}
            >
              <option value="Latest">Latest</option>
              <option value="Relevance">Relevance</option>
              <option value="Rating">Top rated</option>
            </select>
          </label>
          <Button type="submit" variant="secondary" size="sm" disabled={busy}>
            {busy ? <LoaderCircle className="spin" /> : <Search />} Search
          </Button>
        </div>
      </form>
      <div className="library-section-label">
        <span>
          {lastQuery.query
            ? 'Search results'
            : lastQuery.sort === 'Rating'
              ? 'Top rated maps'
              : 'Latest community maps'}
        </span>
        <span>{results.length ? `${results.length} maps` : ''}</span>
      </div>
      {error && (
        <div className="library-message" role="status">
          <p>{error}</p>
          <Button
            variant="outline"
            size="xs"
            onClick={() => void search(lastQuery.query, lastQuery.sort, page)}
          >
            Retry search
          </Button>
        </div>
      )}
      <div className="song-results" aria-busy={busy}>
        {results.map((result) => (
          <article
            className={`song-card${loaded?.id === result.id ? ' is-loaded' : ''}`}
            key={result.id}
          >
            <button
              type="button"
              className="song-card-main"
              disabled={!!loading || !result.difficulties.length}
              onClick={() => choose(result)}
              aria-label={`Load ${result.title} by ${result.artist}`}
            >
              <div className="song-cover">
                <Music2 aria-hidden="true" />
                <img
                  src={result.coverUrl}
                  alt=""
                  loading="lazy"
                  referrerPolicy="no-referrer"
                  onError={(event) => {
                    event.currentTarget.style.display = 'none'
                  }}
                />
              </div>
              <div className="song-card-copy">
                <h3>{result.title}</h3>
                <p>{result.artist}</p>
                <small>Mapped by {result.mapper}</small>
              </div>
              {loaded?.id === result.id ? (
                <Check className="song-load-icon" />
              ) : (
                <ArrowDownToLine className="song-load-icon" />
              )}
            </button>
            <div className="song-card-meta">
              <span>{result.bpm} BPM</span>
              <span>{formatTime(result.duration)}</span>
              <a
                href={`https://beatsaver.com/maps/${result.id}`}
                target="_blank"
                rel="noreferrer"
                aria-label={`Open ${result.title} map source`}
              >
                Map #{result.id}
                <ArrowUpRight aria-hidden="true" />
              </a>
            </div>
            <div className="difficulty-buttons">
              {result.difficulties.map((difficulty, index) => (
                <button
                  type="button"
                  key={`${difficulty.characteristic}-${difficulty.name}-${index}`}
                  disabled={!!loading}
                  data-difficulty={difficulty.name}
                  title={`${difficulty.characteristic} · ${difficulty.notes} notes · ${difficulty.nps.toFixed(1)} notes/sec`}
                  onClick={() => choose(result, difficulty.name, difficulty.characteristic)}
                >
                  {difficultyLabel(difficulty.name)}
                  {difficulty.characteristic !== 'Standard' && (
                    <small> {difficulty.characteristic}</small>
                  )}
                </button>
              ))}
              {!result.difficulties.length && (
                <span className="unsupported-chart">No four-lane chart available</span>
              )}
            </div>
          </article>
        ))}
      </div>
      {busy && (
        <p className="library-loading" role="status">
          <LoaderCircle className="spin" />
          Finding tracks…
        </p>
      )}
      {!busy && !error && searched && !results.length && (
        <div className="library-message">
          <Music2 />
          <p>No maps found. Try an artist, a different spelling, or a map link.</p>
        </div>
      )}
      {more && !error && (
        <Button
          className="load-more"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => void search(lastQuery.query, lastQuery.sort, page + 1)}
        >
          Load more maps
        </Button>
      )}
      <p className="library-footnote">
        Existing charts only. Loaded maps stay in this browser and are cleared on refresh.
      </p>
    </div>
  )
}
