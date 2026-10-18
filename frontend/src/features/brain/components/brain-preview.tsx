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
  Activity,
  ArrowUpRight,
  Brain,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Expand,
  Info,
  Minimize,
  Music2,
  RotateCcw,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SessionOverview } from '@/features/session/components/session-overview'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  cellTypeColor,
  filterEdgeIndices,
  filterNeuronIndices,
  loadBrainData,
  loadBrainEdges,
  loadExplorer,
  neuronConnectionStats,
  topPartnerTypes,
} from '@/features/brain/data'
import type { BrainData, BrainEdges, CellTypeDescription, PartnerType } from '@/features/brain/data'
import { NeuronTrace } from '@/features/brain/components/neuron-trace'
import type { CameraAction } from '@/features/brain/brain-scene'
import { MOVING_EDGE_LIMIT, SAMPLE_EDGE_LIMIT, sampleEdges } from '@/features/brain/rendering'
import type { ConnectionDetail } from '@/features/brain/rendering'
import { SongLibrary } from '@/features/songs/components/song-library'
import { PanelLetter } from '@/components/figure'
import { usePlayer } from '@/features/session/store'
import { liveActivity } from '@/features/brain/live-activity'

const BrainScene = lazy(() => import('@/features/brain/brain-scene'))

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
const partnerText = (partners: PartnerType[], types: string[]) =>
  partners.length
    ? partners.map((p) => `${types[p.type]} (${number.format(p.synapses)})`).join(', ')
    : 'None in this subgraph'
// The camera tip shows until the first drag, zoom or selection, then stays dismissed.
const HINT_KEY = 'fly-saber-brain-hint'
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
  const live = useLiveActivity()
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
  const [showHint, setShowHint] = useState(() => {
    try {
      return localStorage.getItem(HINT_KEY) !== 'seen'
    } catch {
      return true
    }
  })
  const dismissHint = useCallback(() => {
    setShowHint(false)
    try {
      localStorage.setItem(HINT_KEY, 'seen')
    } catch {
      // Storage can be blocked; the tip then returns on the next visit.
    }
  }, [])
  const pickNeuron = useCallback(
    (index: number | null) => {
      setSelected(index)
      if (index !== null) dismissHint()
    },
    [dismissHint],
  )
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

  // What each cell type does (configs/cell_types.yaml). Optional: the view works without it.
  const [explorer, setExplorer] = useState<Map<string, CellTypeDescription> | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    loadExplorer(controller.signal)
      .then((loaded) => {
        if (!controller.signal.aborted) setExplorer(loaded)
      })
      .catch(() => {})
    return () => controller.abort()
  }, [])

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
  const partners = useMemo(
    () => (graph && selected !== null ? topPartnerTypes(graph, selected) : null),
    [graph, selected],
  )
  const about =
    selected !== null && asset && explorer
      ? explorer.get(asset.cell_types[asset.type[selected]])
      : undefined
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
    pickNeuron(indices[(current + direction + indices.length) % indices.length])
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
    if (event.key === 'Escape' && selected !== null) {
      event.preventDefault()
      setSelected(null)
      return
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
        {/* Icons per docs/sidebar-tabs.md (PR #74); the visible label is the name. */}
        <TabsTrigger value="activity">
          <Brain aria-hidden="true" />
          <span>Neurons</span>
        </TabsTrigger>
        <TabsTrigger value="songs">
          <Music2 aria-hidden="true" />
          <span>Songs</span>
        </TabsTrigger>
        <TabsTrigger value="session">
          <Activity aria-hidden="true" />
          <span>Session</span>
        </TabsTrigger>
        <TabsTrigger value="details">
          <Info aria-hidden="true" />
          <span>Details</span>
        </TabsTrigger>
        <TabsTrigger value="next">
          <ArrowUpRight aria-hidden="true" />
          <span>Next steps</span>
        </TabsTrigger>
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
          // The camera tip goes once the viewer drags, scrolls or clicks in the view; the
          // fit-to-view on load moves the camera too, so camera motion alone can't count.
          onPointerDown={showHint ? dismissHint : undefined}
          onWheel={showHint ? dismissHint : undefined}
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
                  onSelect={pickNeuron}
                  onFocus={(index) => navigate.current?.({ focus: index })}
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
          <div className="brain-stage-toolbar" aria-label="Brain camera controls">
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={!indices.length}
              onClick={() => stepSelection(-1)}
              aria-label="Select previous neuron"
              title="Previous neuron"
            >
              <ChevronLeft aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={!indices.length}
              onClick={() => stepSelection(1)}
              aria-label="Select next neuron"
              title="Next neuron"
            >
              <ChevronRight aria-hidden="true" />
            </Button>
            <span className="brain-tool-divider" />
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={!cameraReady}
              aria-label="Reset"
              title="Fit all displayed neurons (Home)"
              onClick={() => navigate.current?.('reset')}
            >
              <RotateCcw aria-hidden="true" />
            </Button>
            <span className="brain-tool-divider" />
            {(['x', 'y', 'z'] as const).map((axis) => (
              <Button
                key={axis}
                variant="ghost"
                size="icon-xs"
                className="axis-button"
                disabled={!cameraReady}
                aria-label={`${axis.toUpperCase()} view`}
                title={`View along display ${axis.toUpperCase()} axis`}
                onClick={() => navigate.current?.(axis)}
              >
                {axis.toUpperCase()}
              </Button>
            ))}
            <span className="brain-tool-divider" />
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={!cameraReady}
              aria-label="Zoom in"
              title="Zoom in (+)"
              onClick={() => navigate.current?.('in')}
            >
              <ZoomIn aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={!cameraReady}
              aria-label="Zoom out"
              title="Zoom out (−)"
              onClick={() => navigate.current?.('out')}
            >
              <ZoomOut aria-hidden="true" />
            </Button>
            <span className="brain-tool-divider" />
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={isFullscreen ? 'Exit neuron view fullscreen' : 'Expand neuron view'}
              onClick={toggleFullscreen}
            >
              {isFullscreen ? <Minimize aria-hidden="true" /> : <Expand aria-hidden="true" />}
            </Button>
          </div>
          {/* Bottom-left: a first-visit tip, then only the selected neuron's card. */}
          <div className="inset-dock">
            {showHint && selected === null && (
              <p className="brain-hint">
                Drag to rotate · right-drag to pan · scroll to zoom · double-click a neuron to focus
              </p>
            )}
            {selected !== null && asset && (
              <div className="neuron-inspector" aria-label="Neuron inspector" role="group">
                <div className="neuron-selection-heading">
                  <h3>
                    <span
                      className="type-swatch"
                      style={{ background: cellTypeColor(asset.type[selected]) }}
                    />
                    {asset.cell_types[asset.type[selected]]}
                  </h3>
                  <div>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      disabled={!cameraReady}
                      onClick={() => navigate.current?.({ focus: selected })}
                      aria-label="Focus camera on this neuron"
                      title="Focus camera on this neuron"
                    >
                      <Crosshair aria-hidden="true" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => setSelected(null)}
                      aria-label="Clear selection"
                      title="Clear selection (Esc)"
                    >
                      <X aria-hidden="true" />
                    </Button>
                  </div>
                </div>
                {about && (
                  <p className="neuron-about">
                    {about.name !== asset.cell_types[asset.type[selected]] && (
                      <strong>{about.name}. </strong>
                    )}
                    {about.description}
                  </p>
                )}
                <NeuronTrace index={selected} />
                {partners && (partners.inputs.length > 0 || partners.outputs.length > 0) && (
                  <dl className="neuron-partners">
                    <div>
                      <dt>Top inputs</dt>
                      <dd>{partnerText(partners.inputs, asset.cell_types)}</dd>
                    </div>
                    <div>
                      <dt>Top outputs</dt>
                      <dd>{partnerText(partners.outputs, asset.cell_types)}</dd>
                    </div>
                  </dl>
                )}
                <dl className="neuron-metadata" aria-live="polite">
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
              </div>
            )}
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
          <div className="brain-view-row">
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
          </div>
          <div className="brain-legend-row">
            <ColorLegend
              swatch={
                selected === null || !asset ? 'var(--ink-3)' : cellTypeColor(asset.type[selected])
              }
            />
            <ActivityNote />
          </div>
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
        </div>
      </TabsContent>
      <TabsContent value="session" className="session-tab">
        <SessionOverview />
      </TabsContent>
      <TabsContent value="details" className="inspector-prose">
        <h2>About the neuron view</h2>
        <dl className="stat-grid">
          <div>
            <dd>{asset ? number.format(asset.n_neurons) : '—'}</dd>
            <dt>Neurons</dt>
          </div>
          <div>
            <dd>{edges ? number.format(edges.n_edges) : '—'}</dd>
            <dt>Directed connections</dt>
          </div>
          <div>
            <dd>{edges ? number.format(edges.n_synapses) : '—'}</dd>
            <dt>Synapses</dt>
          </div>
          <div>
            <dd>{asset ? number.format(asset.cell_types.length) : '—'}</dd>
            <dt>Cell types</dt>
          </div>
        </dl>
        <h3>The points</h3>
        <p>
          Each point is one neuron of the MaleCNS visual-to-motor subgraph, placed at its location,
          not drawn with its branching shape. Some positions are copied from a partner neuron or
          estimated from a cell-type mean; select a neuron to see its position source. The few
          points below the main brain are ascending-neuron somata present in this export.
        </p>
        <h3>The lines</h3>
        <p>
          Lines join presynaptic to postsynaptic neuron locations from the exact graph bundled with
          the trained real fly model (bc_950); they do not trace axon paths. The overview draws an
          evenly spaced sample of 20,000 connections, or every one with Connections: All. A selected
          neuron always shows all its visible connections: incoming in violet, outgoing in gold.
          Filters keep a connection only when both neurons are visible, while the inspector totals
          count every connection.
        </p>
        <h3>Activity and weights</h3>
        <p>
          Synapse counts are anatomy, not learned weights. Training keeps the wiring fixed and
          adjusts neuron and cell-type-pair parameters. During live play, color switches from cell
          type to each neuron&apos;s change from rest: vermilion excited, blue inhibited.
        </p>
        <h3>Limits</h3>
        <p>
          Long axons and a continuous ventral nerve cord would need neuron skeletons or morphology
          data; location points alone cannot reconstruct them.
        </p>
        <dl className="detail-list">
          <div>
            <dt>Activity stream</dt>
            <dd>{live ? 'Live' : 'Starts with live play'}</dd>
          </div>
          <div>
            <dt>Export schema</dt>
            <dd>{asset ? `v${asset.schema_version}` : 'Not loaded'}</dd>
          </div>
        </dl>
        <p className="brain-source-caption">
          {asset?.source ?? 'Source: committed brain export'}. Coordinates are in micrometers. The
          display centers and uniformly scales the data, mapping source (x, y, z) to (x, −z, y).
          Camera presets refer to display axes.
        </p>
      </TabsContent>
      <TabsContent value="next" className="inspector-prose">
        <h2>Roadmap</h2>
        <h3>Working now</h3>
        <ul className="next-steps" data-state="done">
          <li>
            <strong>Live Fly Saber</strong>
            <p>
              Ready-to-play songs run on the server: the brain sees the notes through its eyes and
              drives both sabers, with judged notes and score streamed back.
            </p>
          </li>
          <li>
            <strong>Brain experiments</strong>
            <p>
              Blindfold either eye, lesion the looming detectors and shift arousal from the signal
              path under the game view, mid-song or before it.
            </p>
          </li>
          <li>
            <strong>Live neuron activity</strong>
            <p>Each neuron&apos;s change from rest is drawn on the 3D brain as the fly plays.</p>
          </li>
        </ul>
        <h3>Next</h3>
        <ul className="next-steps">
          <li>
            <strong>Real versus scrambled wiring</strong>
            <p>Play the same song on a brain with shuffled connections to show what wiring buys.</p>
          </li>
          <li>
            <strong>Recorded replay</strong>
            <p>A saved run that plays back when the server is unreachable.</p>
          </li>
        </ul>
      </TabsContent>
    </Tabs>
  )
}
