const SESSION = [
  { term: 'Brain', value: 'None' },
  { term: 'Connectome', value: 'None' },
  { term: 'Map and audio', value: 'None' },
]

export function BrainPanel() {
  return (
    <aside className="brain-panel" aria-labelledby="brain-heading">
      <div className="section-head">
        <h2 id="brain-heading">Neural activity</h2>
        <span className="tag">Illustration</span>
      </div>
      <figure className="brain-view">
        <span className="brain-label brain-label-top">Brain</span>
        <img
          src="/previews/neural-preview.svg"
          className="brain-art"
          alt="Decorative point cloud suggesting a fly brain and ventral nerve cord. Bright points are illustrative, not measured neuron activity."
        />
        <span className="brain-label brain-label-bottom">Ventral nerve cord</span>
        <figcaption className="brain-legend">
          <span>Activity</span>
          <span className="activity-scale" aria-hidden="true">
            Low <i /> High
          </span>
        </figcaption>
      </figure>
      <section className="session-details" aria-labelledby="session-heading">
        <h2 id="session-heading" className="section-title">
          Session
        </h2>
        <dl>
          {SESSION.map(({ term, value }) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </section>
    </aside>
  )
}
