import { useMemo, useRef } from 'react'
import type { CSSProperties } from 'react'
import { Pause, Play, RotateCcw, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { formatTime } from '@/game/chart'
import { usePlayer } from '@/game/store'

export function SessionTimeline() {
  const time = usePlayer((s) => s.time)
  const duration = usePlayer((s) => s.duration)
  const playing = usePlayer((s) => s.playing)
  const chart = usePlayer((s) => s.chart)
  const loading = usePlayer((s) => s.loading)
  const rate = usePlayer((s) => s.rate)
  const volume = usePlayer((s) => s.volume)
  const resumeAfterScrub = useRef(false)
  const scrubbing = useRef(false)
  const lastVolume = useRef(0.7)
  const disabled = !chart || !!loading
  const density = useMemo(() => {
    const bins = Array<number>(96).fill(0)
    if (chart && duration)
      for (const note of chart.notes) {
        if (note.time_sec >= 0 && note.time_sec < duration)
          bins[Math.min(95, Math.floor((note.time_sec / duration) * 96))]++
      }
    const maximum = Math.max(1, ...bins)
    return bins.map((count) => count / maximum)
  }, [chart, duration])

  function finishScrub() {
    if (!scrubbing.current) return
    scrubbing.current = false
    if (resumeAfterScrub.current) void usePlayer.getState().play()
    resumeAfterScrub.current = false
  }

  return (
    <section className="session-timeline" aria-label="Song playback">
      <div className="timeline-controls">
        <div className="playback-buttons">
          <Button
            className="primary-play"
            variant="outline"
            size="icon-sm"
            disabled={disabled}
            onClick={() =>
              playing ? usePlayer.getState().pause() : void usePlayer.getState().play()
            }
            aria-label={playing ? 'Pause song' : 'Play song'}
          >
            {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            disabled={disabled}
            onClick={() => usePlayer.getState().restart()}
            aria-label="Restart song"
          >
            <RotateCcw aria-hidden="true" />
          </Button>
          <select
            className="playback-rate"
            aria-label="Playback speed"
            value={rate}
            onChange={(event) => usePlayer.getState().setRate(Number(event.target.value))}
            disabled={disabled}
          >
            {[0.5, 0.75, 1, 1.25, 1.5].map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
        </div>
        <div className="song-seek" data-loaded={!!chart}>
          <div className="note-density" aria-hidden="true" title="Chart note density">
            {density.map((height, index) => (
              <span
                key={index}
                style={{ height: `${3 + height * 13}px` }}
                data-passed={duration > 0 && index / 96 <= time / duration}
              />
            ))}
          </div>
          <input
            type="range"
            className="song-seek-input"
            aria-label="Song position"
            aria-valuetext={`${formatTime(time)} of ${formatTime(duration)}`}
            min="0"
            max={duration || 1}
            step="0.01"
            value={time}
            disabled={disabled}
            style={{ '--progress': `${duration ? (time / duration) * 100 : 0}%` } as CSSProperties}
            onPointerDown={(event) => {
              if (disabled) return
              event.currentTarget.setPointerCapture(event.pointerId)
              scrubbing.current = true
              resumeAfterScrub.current = usePlayer.getState().playing
              usePlayer.getState().pause()
            }}
            onPointerUp={finishScrub}
            onPointerCancel={finishScrub}
            onLostPointerCapture={finishScrub}
            onBlur={finishScrub}
            onChange={(event) => usePlayer.getState().seek(Number(event.target.value))}
          />
          <div className="song-seek-ticks" aria-hidden="true">
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
              <span key={fraction}>
                {chart ? formatTime(duration * fraction) : fraction === 0 ? '00:00' : '--:--'}
              </span>
            ))}
          </div>
        </div>
        <div className="song-time">
          <span>{formatTime(time)}</span>
          <span>/ {chart ? formatTime(duration) : '--:--'}</span>
        </div>
        <div className="song-volume">
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={volume ? 'Mute song' : 'Unmute song'}
            onClick={() => {
              if (volume) {
                lastVolume.current = volume
                usePlayer.getState().setVolume(0)
              } else usePlayer.getState().setVolume(lastVolume.current)
            }}
          >
            {volume ? <Volume2 aria-hidden="true" /> : <VolumeX aria-hidden="true" />}
          </Button>
          <input
            type="range"
            aria-label="Song volume"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={(event) => usePlayer.getState().setVolume(Number(event.target.value))}
          />
        </div>
      </div>
    </section>
  )
}
