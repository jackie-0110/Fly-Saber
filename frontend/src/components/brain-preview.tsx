import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import {
  Brain,
  ChevronLeft,
  ChevronRight,
  Expand,
  Minimize,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SessionOverview } from '@/components/session-overview'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  cellTypeColor,
  filterEdgeIndices,
  filterNeuronIndices,
  loadBrainData,
  loadBrainEdges,
  neuronConnectionStats,
} from '@/brain/data'
import type { BrainData, BrainEdges } from '@/brain/data'
import type { CameraAction } from '@/brain/brain-scene'
import { MOVING_EDGE_LIMIT, SAMPLE_EDGE_LIMIT, sampleEdges } from '@/brain/rendering'
import type { ConnectionDetail } from '@/brain/rendering'
import { SongLibrary } from '@/components/song-library'
import { PanelLetter } from '@/components/figure'
import { usePlayer } from '@/game/store'
import { liveActivity } from '@/brain/live-activity'

const BrainScene = lazy(() => import('@/brain/brain-scene'))

function useLiveActivity() {
  return useSyncExternalStore(
    (notify) => liveActivity.subscribe(notify),
    () => liveActivity.live(),
  )
}

function ActivityNote() {
  const live = useLiveActivity()
  return (
    <p className="brain-backend-note">
      <span className={`status-dot ${live ? 'green' : 'amber'}`} />
      <span>
        {live ? 'Live: activity is change from rest.' : 'Activity shows during live play.'}
      </span>
    </p>
  )
}

/**
 * Idle, color is cell type. Live, color is the sign of activity, so the same row shows
 * the diverging scale instead (one row either way, so the view never resizes).
 */
function ColorLegend({ swatch }: { swatch: string }) {
  const live = useLiveActivity()
  return (
    <div className="brain-color-legend">
      {live ? (
        <span className="activity-scale">
          <span>Inhibited</span>
          <i aria-hidden="true" />
          <span>Excited</span>
        </span>
      ) : (
        <>
          <span className="type-swatch" style={{ background: swatch }} />
          <span>Color: cell type</span>
        </>
      )}
    </div>
  )
}
const number = new Intl.NumberFormat('en-US')
const provenance: Record<string, string> = {
  soma: 'Soma position',
  partner: 'Copied from strongest partner',
  type_mean: 'Estimated from cell-type mean or overall mean',
}

class SceneBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function SceneFallback({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="brain-fallback" role="status">
      <Brain aria-hidden="true" />
      <p>{message}</p>
      {onRetry && (
        <Button variant="outline" size="xs" onClick={onRetry}>
          Retry 3D view
        </Button>
      )}
    </div>
  )
}

export function BrainPreview() {
  const inspectorTab = usePlayer((state) => state.inspectorTab)
  const setInspectorTab = usePlayer((state) => state.setInspectorTab)
  const [data, setData] = useState<BrainData | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [sceneAttempt, setSceneAttempt] = useState(0)
  const [contextLost, setContextLost] = useState(false)
  const [group, setGroup] = useState('')
  const [cellType, setCellType] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const navigate = useRef<((action: CameraAction) => void) | null>(null)
  const [cameraReady, setCameraReady] = useState(false)
  const [pointSize, setPointSize] = useState(1.5)
  const [opacity, setOpacity] = useState(0.65)
  const [detail, setDetail] = useState<ConnectionDetail>('sample')
  const [edges, setEdges] = useState<BrainEdges | null>(null)
  const [edgesError, setEdgesError] = useState('')
  const [edgesAttempt, setEdgesAttempt] = useState(0)
  const showConnections = detail !== 'off'
  const [cameraMoving, setCameraMoving] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [fullscreenError, setFullscreenError] = useState('')
  const onNavigateReady = useCallback((handler: ((action: CameraAction) => void) | null) => {
    navigate.current = handler
    setCameraReady(handler !== null)
  }, [])
  const onContextLost = useCallback(() => {
    setContextLost(true)
    setCameraReady(false)
  }, [])

  useEffect(() => {
    const updateFullscreen = () =>
      setIsFullscreen(document.fullscreenElement === stageRef.current && stageRef.current !== null)
    document.addEventListener('fullscreenchange', updateFullscreen)
    return () => document.removeEventListener('fullscreenchange', updateFullscreen)
  }, [])

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement === stageRef.current) await document.exitFullscreen()
      else await stageRef.current?.requestFullscreen()
      setFullscreenError('')
    } catch {
      setFullscreenError(
        'Fullscreen is unavailable in this browser. Use zoom to inspect the cloud.',
      )
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    loadBrainData(controller.signal)
      .then((loaded) => {
        if (!controller.signal.aborted) setData(loaded)
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : 'Could not load brain data.')
      })
    return () => controller.abort()
  }, [attempt])

  // Fetch the connection table only once connections are shown, after the neurons render.
  const asset = data?.asset
  const wantEdges = !!asset && showConnections && !edges
  useEffect(() => {
    if (!wantEdges || !asset) return
    const controller = new AbortController()
    loadBrainEdges(asset, controller.signal)
      .then((loaded) => {
        if (!controller.signal.aborted) setEdges(loaded)
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setEdgesError(cause instanceof Error ? cause.message : 'Could not load connections.')
      })
    return () => controller.abort()
  }, [asset, wantEdges, edgesAttempt])

  // `data` keeps its identity when edges arrive, so the camera is fitted only once.
  const graph = useMemo(() => (data && edges ? { ...data, edges } : null), [data, edges])
  const indices = useMemo(
    () => (data ? filterNeuronIndices(data, group, cellType) : []),
    [data, group, cellType],
  )
  const filteredEdges = useMemo(
    () => (graph && showConnections ? filterEdgeIndices(graph, indices, selected) : []),
    [graph, indices, selected, showConnections],
  )
  // A selected neuron always shows every visible connection; sampling is for the overview.
  const edgeIndices = useMemo(
    () =>
      detail === 'sample' && selected === null
        ? sampleEdges(filteredEdges, SAMPLE_EDGE_LIMIT)
        : filteredEdges,
    [detail, filteredEdges, selected],
  )
  const connections = useMemo(
    () => (graph && selected !== null ? neuronConnectionStats(graph, selected) : null),
    [graph, selected],
  )
  const availableCellTypes = useMemo(() => {
    if (!data) return []
    const members = filterNeuronIndices(data, group, '')
    return [...new Set(members.map((index) => data.asset.type[index]))].sort((a, b) => a - b)
  }, [data, group])
  const selectGroup = (value: string) => {
    setGroup(value)
    if (
      data &&
      cellType !== '' &&
      !filterNeuronIndices(data, value, '').some(
        (index) => data.asset.type[index] === Number(cellType),
      )
    )
      setCellType('')
    setSelected(null)
  }
  const selectType = (value: string) => {
    setCellType(value)
    setSelected(null)
  }
  const retryScene = () => {
    setContextLost(false)
    setSceneAttempt((value) => value + 1)
  }
  const stepSelection = (direction: number) => {
    if (!indices.length) return
    const current = selected === null ? (direction > 0 ? -1 : 0) : indices.indexOf(selected)
    setSelected(indices[(current + direction + indices.length) % indices.length])
  }
  const onCameraKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return
    const action: Record<string, CameraAction> = {
      ArrowLeft: 'left',
      ArrowRight: 'right',
      ArrowUp: 'up',
      ArrowDown: 'down',
      '+': 'in',
      '=': 'in',
      '-': 'out',
      Home: 'reset',
    }
    if (action[event.key]) {
      event.preventDefault()
      navigate.current?.(action[event.key])
    }
  }
  const fallback = (
    <SceneFallback
      message="The 3D view is unavailable. Your browser needs WebGL 2. You can still explore neuron metadata below."
      onRetry={retryScene}
    />
  )

  return (
    <Tabs
      value={inspectorTab}
      onValueChange={(value) => setInspectorTab(value as typeof inspectorTab)}
      className="brain-inspector"
    >
      <TabsList className="inspector-tabs" aria-label="Brain inspector view">
        <TabsTrigger value="activity">Neurons</TabsTrigger>
        <TabsTrigger value="songs">Songs</TabsTrigger>
        <TabsTrigger value="session">Session</TabsTrigger>
        <TabsTrigger value="details">Details</TabsTrigger>
        <TabsTrigger value="next">Next steps</TabsTrigger>
      </TabsList>
      <TabsContent value="songs" className="songs-tab" forceMount>
        <SongLibrary />
      </TabsContent>
      <TabsContent value="activity" className="activity-tab">
        <div
          className="brain-stage brain-stage-3d"
          ref={stageRef}
          tabIndex={0}
          role="region"
          aria-label="Interactive 3D fly neuron view"
          onKeyDown={onCameraKey}
        >
          <div className="inset-label">
            <PanelLetter>b</PanelLetter>
            <div>
              <h2>Fly brain neurons</h2>
              <p>MaleCNS v1.0 subgraph</p>
            </div>
          </div>
          {error ? (
            <SceneFallback
              message={error}
              onRetry={() => {
                setError('')
                setAttempt((value) => value + 1)
              }}
            />
          ) : !data ? (
            <SceneFallback message="Loading neuron coordinates and connections…" />
          ) : contextLost ? (
            fallback
          ) : (
            <SceneBoundary key={sceneAttempt} fallback={fallback}>
              <Suspense fallback={<SceneFallback message="Preparing the 3D view…" />}>
                <BrainScene
                  data={data}
                  edges={edges}
                  indices={indices}
                  edgeIndices={edgeIndices}
                  showConnections={showConnections}
                  selected={selected}
                  onSelect={setSelected}
                  onNavigateReady={onNavigateReady}
                  onFailure={onContextLost}
                  pointSize={pointSize}
                  opacity={opacity}
                  onMotionChange={setCameraMoving}
                />
              </Suspense>
            </SceneBoundary>
          )}
          {data && !indices.length && <p className="brain-empty">No neurons match both filters.</p>}
          <div className="brain-stage-toolbar">
            <div>
              <Button
                variant="ghost"
                size="icon-xs"
                disabled={!cameraReady}
                aria-label="Fit all displayed neurons"
                onClick={() => navigate.current?.('reset')}
              >
                <RotateCcw aria-hidden="true" />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={isFullscreen ? 'Exit neuron view fullscreen' : 'Expand neuron view'}
                onClick={toggleFullscreen}
              >
                {isFullscreen ? <Minimize aria-hidden="true" /> : <Expand aria-hidden="true" />}
              </Button>
            </div>
          </div>
          <div className="inset-dock">
            <div className="neuron-inspector">
              <div className="neuron-selection-heading">
                <h3>Neuron inspector</h3>
                <div>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    disabled={!indices.length}
                    onClick={() => stepSelection(-1)}
                    aria-label="Select previous neuron"
                  >
                    <ChevronLeft aria-hidden="true" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    disabled={!indices.length}
                    onClick={() => stepSelection(1)}
                    aria-label="Select next neuron"
                  >
                    <ChevronRight aria-hidden="true" />
                  </Button>
                </div>
              </div>
              {selected === null || !asset ? (
                <p>
                  Click a neuron or use the previous/next buttons to inspect its identity and
                  position.
                </p>
              ) : (
                <dl className="neuron-metadata" aria-live="polite">
                  <div>
                    <dt>Cell type</dt>
                    <dd>{asset.cell_types[asset.type[selected]]}</dd>
                  </div>
                  <div>
                    <dt>Body ID / index</dt>
                    <dd>
                      {String(asset.body_id[selected])} / {selected}
                    </dd>
                  </div>
                  <div>
                    <dt>Side / role</dt>
                    <dd>
                      {asset.sides[asset.side[selected]]} / {asset.roles[asset.role[selected]]}
                    </dd>
                  </div>
                  <div>
                    <dt>Transmitter</dt>
                    <dd>{asset.nts[asset.nt[selected]]}</dd>
                  </div>
                  <div>
                    <dt>Class</dt>
                    <dd>{asset.superclasses[asset.superclass[selected]].replaceAll('_', ' ')}</dd>
                  </div>
                  {connections && (
                    <>
                      <div>
                        <dt>Incoming</dt>
                        <dd>
                          {number.format(connections.incoming)} connections /{' '}
                          {number.format(connections.incomingSynapses)} synapses
                        </dd>
                      </div>
                      <div>
                        <dt>Outgoing</dt>
                        <dd>
                          {number.format(connections.outgoing)} connections /{' '}
                          {number.format(connections.outgoingSynapses)} synapses
                        </dd>
                      </div>
                    </>
                  )}
                  <div>
                    <dt>Position (µm)</dt>
                    <dd>
                      {asset.position_um
                        .slice(selected * 3, selected * 3 + 3)
                        .map((v) => v.toFixed(1))
                        .join(', ')}
                    </dd>
                  </div>
                  <div>
                    <dt>Position source</dt>
                    <dd>
                      {provenance[asset.position_sources[asset.position_source[selected]]] ??
                        asset.position_sources[asset.position_source[selected]]}
                    </dd>
                  </div>
                </dl>
              )}
            </div>
            <div className="brain-dock">
              <div className="brain-camera-controls" aria-label="Brain camera controls">
                <Button
                  variant="ghost"
                  size="xs"
                  disabled={!cameraReady}
                  onClick={() => navigate.current?.('reset')}
                  title="Reset camera (Home)"
                >
                  <RotateCcw aria-hidden="true" />
                  Reset
                </Button>
                {(['x', 'y', 'z'] as const).map((axis) => (
                  <Button
                    key={axis}
                    variant="ghost"
                    size="xs"
                    disabled={!cameraReady}
                    onClick={() => navigate.current?.(axis)}
                    title={`View along display ${axis.toUpperCase()} axis`}
                  >
                    {axis.toUpperCase()} view
                  </Button>
                ))}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  disabled={!cameraReady}
                  aria-label="Zoom in"
                  onClick={() => navigate.current?.('in')}
                >
                  <ZoomIn aria-hidden="true" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  disabled={!cameraReady}
                  aria-label="Zoom out"
                  onClick={() => navigate.current?.('out')}
                >
                  <ZoomOut aria-hidden="true" />
                </Button>
              </div>
              <div className="brain-display-controls">
                <label>
                  Connections
                  <select
                    aria-label="Connection detail"
                    value={detail}
                    disabled={!data}
                    onChange={(event) => setDetail(event.target.value as ConnectionDetail)}
                  >
                    <option value="off">Off</option>
                    <option value="sample">Sample</option>
                    <option value="all">All</option>
                  </select>
                </label>
                <label>
                  Point size
                  <input
                    aria-label="Neuron point size"
                    type="range"
                    min="1"
                    max="5"
                    step="0.5"
                    value={pointSize}
                    onChange={(event) => setPointSize(Number(event.target.value))}
                  />
                </label>
                <label>
                  Opacity
                  <input
                    aria-label="Neuron point opacity"
                    type="range"
                    min="0.15"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(event) => setOpacity(Number(event.target.value))}
                  />
                </label>
              </div>
            </div>
          </div>
        </div>
        {fullscreenError && (
          <p className="brain-camera-help" role="status">
            {fullscreenError}
          </p>
        )}
        <div className="brain-details">
          <div className="brain-filters">
            <label htmlFor="brain-group">
              Neuron group
              <select
                id="brain-group"
                aria-label="Neuron group"
                value={group}
                disabled={!data}
                onChange={(event) => selectGroup(event.target.value)}
              >
                <option value="">All groups</option>
                {data &&
                  Object.entries(data.groups).map(([name, members]) => (
                    <option key={name} value={name}>
                      {name.replaceAll('_', ' ')} ({number.format(members.length)})
                    </option>
                  ))}
              </select>
            </label>
            <label htmlFor="brain-type">
              Cell type
              <select
                id="brain-type"
                aria-label="Cell type"
                value={cellType}
                disabled={!data}
                onChange={(event) => selectType(event.target.value)}
              >
                <option value="">All cell types</option>
                {availableCellTypes.map((index) => (
                  <option key={index} value={index}>
                    {asset?.cell_types[index]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="neuron-count" role="status">
            {asset
              ? `${number.format(indices.length)} of ${number.format(asset.n_neurons)} neurons shown`
              : 'Loading neuron coordinates and connections…'}
            {(group || cellType) && (
              <Button
                variant="ghost"
                size="xs"
                onClick={() => {
                  selectGroup('')
                  selectType('')
                }}
              >
                Clear filters
              </Button>
            )}
          </div>
          <ColorLegend
            swatch={
              selected === null || !asset ? 'var(--ink-3)' : cellTypeColor(asset.type[selected])
            }
          />
          <div className="brain-connection-summary" role="status">
            <span>
              {!data || !showConnections
                ? showConnections
                  ? 'Loading connections…'
                  : 'Connections hidden'
                : edgesError
                  ? edgesError
                  : !edges
                    ? 'Loading connections…'
                    : `${number.format(cameraMoving && detail === 'all' && selected === null ? Math.min(edgeIndices.length, MOVING_EDGE_LIMIT) : edgeIndices.length)} of ${number.format(edges.n_edges)} connections shown`}
            </span>
            {edgesError && showConnections && (
              <Button
                variant="outline"
                size="xs"
                onClick={() => {
                  setEdgesError('')
                  setEdgesAttempt((value) => value + 1)
                }}
              >
                Retry connections
              </Button>
            )}
            {edges && showConnections && selected !== null && (
              <span>
                <span className="edge-in">Violet: incoming</span>
                {' · '}
                <span className="edge-out">Gold: outgoing</span>
              </span>
            )}
            {edges && showConnections && selected === null && (
              <span>
                {cameraMoving && detail === 'all' && edgeIndices.length > MOVING_EDGE_LIMIT
                  ? 'Movement preview.'
                  : detail === 'sample' && filteredEdges.length > SAMPLE_EDGE_LIMIT
                    ? 'Evenly spaced sample.'
                    : 'Select a neuron to trace its connections.'}
              </span>
            )}
          </div>
          <ActivityNote />
        </div>
      </TabsContent>
      <TabsContent value="session" className="session-tab">
        <SessionOverview />
      </TabsContent>
      <TabsContent value="details" className="inspector-prose">
        <h2>About the neuron view</h2>
        <p>
          This view uses the supplied MaleCNS visual-to-motor subgraph. The few points below the
          main brain are ascending-neuron soma locations present in this export. It is a subset of
          the source connectome.
        </p>
        <p>
          Connections come from the exact graph bundled with the trained real fly model (bc_950,
          training update 950). Lines join presynaptic to postsynaptic neuron locations; they do not
          trace axon paths. By default the overview draws an evenly spaced sample of 20,000 real
          connections; choose All to draw every connection. With All, camera movement draws the
          sample at a lower pixel resolution and full detail returns when the camera settles.
          Selected-neuron connections are always complete. Filters keep connections only when both
          neurons are visible; selecting a neuron shows its visible incoming (blue) and outgoing
          (gold) connections. Inspector totals include all its connections, including those hidden
          by filters. Overlapping positions can hide individual lines.
        </p>
        <p>
          Synapse counts describe anatomical connections, not learned weights or firing activity.
          Training keeps the wiring fixed and adjusts neuron and cell-type-pair parameters.
        </p>
        <p>
          Points summarize neuron locations, not their full branching shapes. Some positions are
          copied from a partner neuron or estimated from a cell-type mean. Select a neuron to see
          its position source.
        </p>
        <p>
          Long axons and a continuous ventral nerve cord shape require additional neuron skeletons
          or morphology data. Those structures cannot be reconstructed from these location points
          alone.
        </p>
        <dl className="detail-list">
          <div>
            <dt>Neuron count</dt>
            <dd>{asset ? number.format(asset.n_neurons) : 'Not loaded'}</dd>
          </div>
          <div>
            <dt>Directed connections</dt>
            <dd>{edges ? number.format(edges.n_edges) : 'Not loaded'}</dd>
          </div>
          <div>
            <dt>Synapses</dt>
            <dd>{edges ? number.format(edges.n_synapses) : 'Not loaded'}</dd>
          </div>
          <div>
            <dt>Cell types</dt>
            <dd>{asset ? number.format(asset.cell_types.length) : 'Not loaded'}</dd>
          </div>
          <div>
            <dt>Export schema</dt>
            <dd>{asset ? `v${asset.schema_version}` : 'Not loaded'}</dd>
          </div>
          <div>
            <dt>Activity stream</dt>
            <dd>Awaiting backend</dd>
          </div>
        </dl>
        <p className="brain-source-caption">
          {asset?.source ?? 'Source: committed brain export'}. Coordinates are in micrometers. The
          display centers and uniformly scales the data, mapping source (x, y, z) to (x, −z, y).
          Camera presets refer to display axes.
        </p>
      </TabsContent>
      <TabsContent value="next" className="inspector-prose">
        <h2>Planned features</h2>
        <ul className="next-steps">
          <li>
            <strong>Real-time Fly Saber</strong>
            <p>
              Live songs connect the song clock to server saber actions, judged notes and scores.
              Choose a server song and player in the Songs tab to start.
            </p>
          </li>
          <li>
            <strong>Live experiments</strong>
            <p>Lesion controls, real versus scrambled wiring, and a recorded replay fallback.</p>
          </li>
        </ul>
      </TabsContent>
    </Tabs>
  )
}
