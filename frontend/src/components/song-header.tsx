import { ArrowUpRight, Music2 } from 'lucide-react'
import { difficultyLabel, formatTime } from '@/game/chart'
import { usePlayer } from '@/game/store'

export function SongHeader() {
  const song = usePlayer((s) => s.song)
  const chart = usePlayer((s) => s.chart)
  const coverUrl = usePlayer((s) => s.coverUrl)
  const duration = usePlayer((s) => s.duration)
  return (
    <section className="song-header" aria-label="Selected song information">
      <div className="current-song-cover">
        <Music2 aria-hidden="true" />
        {coverUrl && <img src={coverUrl} alt="" />}
      </div>
      <div className="current-song-copy">
        <div className="current-song-title">
          <h1>{song?.title ?? 'No song selected'}</h1>
          {chart && (
            <span className="current-difficulty" data-difficulty={chart.difficulty.name}>
              {difficultyLabel(chart.difficulty.name)}
            </span>
          )}
        </div>
        {song && chart && (
          <div className="current-song-meta">
            <span>
              {song.artist}
              {song.subtitle && ` · ${song.subtitle}`}
            </span>
            <span>{song.bpm} BPM</span>
            <span>{formatTime(duration)}</span>
            <span>{chart.notes.filter((n) => !n.is_bomb).length.toLocaleString()} notes</span>
            <span>Mapped by {song.mapper}</span>
            {song.serverMapId && (
              <span>{song.split === 'heldout' ? 'Unseen map' : 'Training map'} · Live play</span>
            )}
            {song.sourceUrl && (
              <a href={song.sourceUrl} target="_blank" rel="noreferrer">
                Map source
                <ArrowUpRight aria-hidden="true" />
              </a>
            )}
          </div>
        )}
      </div>
      {song?.attribution && <p className="song-attribution">{song.attribution}</p>}
    </section>
  )
}
