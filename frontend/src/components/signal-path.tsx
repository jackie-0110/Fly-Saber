import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-react'

// Ghost hex lattice for the two compound eyes. Shape only: the live stream fills the
// brain's 437 eye columns (docs/PROTOCOL.md, section 2).
const OMMATIDIA = (() => {
  const points: [number, number][] = []
  const spacing = 4.2
  const rowHeight = spacing * 0.866
  for (const cx of [42, 118]) {
    for (let row = -11; row <= 11; row++) {
      const y = row * rowHeight
      const offset = row % 2 ? spacing / 2 : 0
      for (let col = -9; col <= 9; col++) {
        const x = col * spacing + offset
        if ((x / 30) ** 2 + (y / 37) ** 2 <= 1) points.push([cx + x, 45 + y])
      }
    }
  }
  return points
})()

// Outline heights for the descending-neuron skeleton chart.
const DN_BARS = [
  14, 22, 9, 30, 18, 12, 26, 8, 20, 34, 15, 11, 24, 17, 28, 10, 19, 13, 31, 16, 9, 23,
]

const SABERS = [
  { name: 'Left saber', tone: 'coral' },
  { name: 'Right saber', tone: 'mint' },
]
const CUTS = [ArrowUp, ArrowDown, ArrowLeft, ArrowRight]

export function SignalPath() {
  return (
    <section className="signal-path" aria-labelledby="signal-heading">
      <h2 id="signal-heading" className="sr-only">
        Signal path
      </h2>
      <ol className="signal-stages">
        <li className="stage">
          <StageHead step={1} title="Visual input" caption="437 ommatidial columns, both eyes" />
          <svg
            className="eye-lattice"
            viewBox="0 0 160 90"
            preserveAspectRatio="xMinYMid meet"
            aria-hidden="true"
          >
            {OMMATIDIA.map(([x, y]) => (
              <circle key={`${x}-${y}`} cx={x} cy={y} r="1.15" />
            ))}
          </svg>
        </li>
        <li className="stage-link" aria-hidden="true">
          <ArrowRight />
        </li>
        <li className="stage">
          <StageHead
            step={2}
            title="Descending neurons"
            caption="Brain output to the motor system"
          />
          <svg
            className="dn-skeleton"
            viewBox="0 0 176 52"
            preserveAspectRatio="xMinYMid meet"
            aria-hidden="true"
          >
            {DN_BARS.map((h, i) => (
              <rect key={i} x={i * 8 + 1} y={50 - h} width="5" height={h} rx="1" />
            ))}
            <line x1="0" x2="176" y1="50.5" y2="50.5" />
          </svg>
        </li>
        <li className="stage-link" aria-hidden="true">
          <ArrowRight />
        </li>
        <li className="stage">
          <StageHead step={3} title="Saber action" caption="Target cell and cut direction" />
          <div className="motor-skeleton" aria-hidden="true">
            {SABERS.map(({ name, tone }) => (
              <div key={name} className="motor-row">
                <span className="motor-name">
                  <i className={`swatch ${tone}`} />
                  {name}
                </span>
                <span className="cell-grid">
                  {Array.from({ length: 12 }, (_, i) => (
                    <i key={i} />
                  ))}
                </span>
                <span className="cut-set">
                  {CUTS.map((Cut, i) => (
                    <Cut key={i} />
                  ))}
                </span>
              </div>
            ))}
          </div>
        </li>
      </ol>
    </section>
  )
}

function StageHead({ step, title, caption }: { step: number; title: string; caption: string }) {
  return (
    <div className="stage-head">
      <span className="stage-step" aria-hidden="true">
        {step}
      </span>
      <div>
        <h3>{title}</h3>
        <p>{caption}</p>
      </div>
    </div>
  )
}
