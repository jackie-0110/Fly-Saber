import { usePlayer } from '@/game/store'
import { useSession } from '@/game/session'

export function PanelLetter({ children }: { children: string }) {
  return (
    <span className="panel-letter" aria-hidden="true">
      {children}
    </span>
  )
}

export function FigureCaption() {
  const song = usePlayer((s) => s.song)
  const chart = usePlayer((s) => s.chart)
  const status = useSession((s) => s.status)
  return (
    <footer className="figure-caption">
      <p>
        <strong>Fig. 1 | Closed-loop visuomotor control of Fly Saber.</strong> <b>a</b>, Rhythm task
        {song && chart ? (
          <>
            {' '}
            (<i>{song.title}</i>, {chart.difficulty.name})
          </>
        ) : null}
        ; sabers are keyed <span className="key-green">green</span> (right) and{' '}
        <span className="key-magenta">magenta</span> (left). <b>b</b>, MaleCNS v1.0 visual-to-motor
        subgraph, one point per neuron at its soma position (µm), coloured by cell type. <b>c</b>,
        Note density over song time. <b>d</b>, Readouts ordered from stimulus to behaviour.{' '}
        <span className="caption-status">
          {status === 'idle'
            ? 'No live session; panels show the chart only.'
            : `Live session: ${status}.`}
        </span>
      </p>
    </footer>
  )
}
