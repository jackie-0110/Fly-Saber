import { compareScores } from '@/features/session/comparison'
import type { Comparison } from '@/features/session/comparison'
import { useBrainControls } from '@/features/session/controls-state'
import { isUntouched } from '@/features/session/interventions'
import { useSession } from '@/features/session/session'

/*
 * The baseline for every brain control: an untouched twin (the same brain on the same song
 * and seed, with no controls) plays beside the fly, so the gap in hit rate is what the
 * controls did. With nothing changed the two play note for note alike.
 */

// Under this many judged notes the gap is mostly luck, so the copy says so.
const FEW_NOTES = 20
// Gaps smaller than this read as "no difference".
const SAME_POINTS = 0.5

const percent = (rate: number) => `${(rate * 100).toFixed(1)}%`
const signedPoints = (points: number) =>
  `${points > 0 ? '+' : points < 0 ? '−' : '±'}${Math.abs(points).toFixed(1)} pts`

function useComparison() {
  const score = useSession((s) => s.score)
  const twin = useSession((s) => s.baselineScore)
  const status = useSession((s) => s.status)
  const compare = useBrainControls((s) => s.compare)
  const touched = !isUntouched(useBrainControls((s) => s.controls))
  return { comparison: compareScores(score, twin), status, compare, touched }
}

function tone(comparison: Comparison) {
  if (Math.abs(comparison.points) < SAME_POINTS) return 'same'
  return comparison.points > 0 ? 'better' : 'worse'
}

function verdict(comparison: Comparison, touched: boolean) {
  if (!touched) return 'No controls changed, so both play alike.'
  const few =
    comparison.judged < FEW_NOTES ? ` Only ${comparison.judged} notes in: early days.` : ''
  if (tone(comparison) === 'same') return `The controls make no clear difference yet.${few}`
  return `${tone(comparison) === 'better' ? 'Better' : 'Worse'} than the untouched brain by ${Math.abs(comparison.points).toFixed(1)} points.${few}`
}

/** The comparison with its switch, for the Brain controls window and the Session tab. */
export function BaselineCompare() {
  const { comparison, status, compare, touched } = useComparison()
  const setCompare = useBrainControls((s) => s.setCompare)
  const live = ['playing', 'paused', 'ended'].includes(status)

  return (
    <section className="baseline-compare" aria-labelledby="baseline-heading">
      <div className="baseline-heading">
        <h3 id="baseline-heading">Against an untouched brain</h3>
        <button
          type="button"
          className="preset-button"
          aria-pressed={compare}
          onClick={() => setCompare(!compare)}
        >
          Compare
        </button>
      </div>
      {!compare ? (
        <p className="baseline-note">
          Off. Turn on to run an untouched twin of this fly on the same notes, as a baseline for
          your controls.
        </p>
      ) : comparison ? (
        <>
          <dl className="baseline-bars" data-tone={tone(comparison)}>
            <BaselineBar label="Your fly" rate={comparison.fly} kind="fly" />
            <BaselineBar label="Untouched" rate={comparison.twin} kind="twin" />
          </dl>
          <p className="baseline-note" role="status">
            <strong className="baseline-delta" data-tone={tone(comparison)}>
              {signedPoints(comparison.points)}
            </strong>{' '}
            {verdict(comparison, touched)}
          </p>
        </>
      ) : (
        <p className="baseline-note">
          {live
            ? 'Waiting for the first judged notes.'
            : 'An untouched twin plays the same song beside your fly. Play a song to see what your controls change.'}
        </p>
      )}
    </section>
  )
}

function BaselineBar({ label, rate, kind }: { label: string; rate: number; kind: string }) {
  return (
    <div className="baseline-bar" data-kind={kind}>
      <dt>{label}</dt>
      <dd>
        <span className="baseline-track" aria-hidden="true">
          <span className="baseline-fill" style={{ width: `${rate * 100}%` }} />
        </span>
        <span className="baseline-value">{percent(rate)}</span>
      </dd>
    </div>
  )
}

/** One line under Performance: the twin's accuracy and the gap. */
export function BaselineReadout() {
  const { comparison, compare } = useComparison()
  if (!compare || !comparison) return null
  return (
    <span className="readout-caption baseline-readout">
      Untouched {percent(comparison.twin)} ·{' '}
      <span className="baseline-delta" data-tone={tone(comparison)}>
        {signedPoints(comparison.points)}
      </span>
    </span>
  )
}
