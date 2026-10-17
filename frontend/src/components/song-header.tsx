import { ArrowUpRight, ChevronDown, Music2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { difficultyLabel, formatTime } from '@/game/chart'
import { openSongLibrary, usePlayer } from '@/game/store'

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
          <h1>{song?.title ?? 'Your next track starts here.'}</h1>
          {chart && (
            <span className="current-difficulty" data-difficulty={chart.difficulty.name}>
              {difficultyLabel(chart.difficulty.name)}
            </span>
          )}
        </div>
        {song && chart ? (
          <div className="current-song-meta">
            <span>
              {song.artist}
              {song.subtitle && ` · ${song.subtitle}`}
            </span>
            <span>{song.bpm} BPM</span>
            <span>{formatTime(duration)}</span>
            <span>{chart.notes.filter((n) => !n.is_bomb).length.toLocaleString()} notes</span>
            <span>Mapped by {song.mapper}</span>
            {song.sourceUrl && (
              <a href={song.sourceUrl} target="_blank" rel="noreferrer">
                Map source
                <ArrowUpRight aria-hidden="true" />
              </a>
            )}
          </div>
        ) : (
          <p>Choose a community song or import a map to explore it in 3D.</p>
        )}
      </div>
      <Button className="choose-song" variant="outline" size="sm" onClick={openSongLibrary}>
        <Music2 aria-hidden="true" />
        {song ? 'Change song' : 'Choose song'}
        <ChevronDown aria-hidden="true" />
      </Button>
    </section>
  )
}
