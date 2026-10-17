import { Brain, Orbit, RotateCcw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function BrainPreview() {
  return (
    <Tabs defaultValue="activity" className="brain-inspector">
      <TabsList className="inspector-tabs" aria-label="Brain inspector view">
        <TabsTrigger value="activity">Activity</TabsTrigger>
        <TabsTrigger value="details">Details</TabsTrigger>
        <TabsTrigger value="next">Next steps</TabsTrigger>
      </TabsList>
      <TabsContent value="activity" className="activity-tab">
        <div className="brain-title">
          <div>
            <Brain aria-hidden="true" />
            <h2>Neural activity</h2>
          </div>
          <Badge variant="outline">Preview</Badge>
        </div>
        <p className="brain-subtitle">Spatial illustration / no live data</p>
        <div className="brain-stage">
          <div className="brain-annotation">
            <span className="annotation-line" />
            <span>BRAIN</span>
          </div>
          <img
            src="/previews/neural-preview.svg"
            className="brain-art"
            alt="Decorative point cloud suggesting a fly brain and ventral nerve cord. Bright points are illustrative, not measured neuron activity."
          />
          <div className="cord-annotation">
            <span className="annotation-line" />
            <span>VENTRAL NERVE CORD</span>
          </div>
          <div className="axis-marker" aria-hidden="true">
            <span className="axis-y">Y</span>
            <span className="axis-z">Z</span>
            <span className="axis-x">X</span>
            <svg viewBox="0 0 48 48">
              <path d="M18 29V6m0 23 23 8M18 29 5 40" fill="none" stroke="currentColor" />
            </svg>
          </div>
        </div>
        <div className="brain-legend">
          <span>Illustrative activity</span>
          <div className="activity-scale">
            <span>Low</span>
            <i />
            <span>High</span>
          </div>
        </div>
        <div className="rotation-note">
          <Orbit aria-hidden="true" />
          <span>3D orbit and zoom coming next</span>
          <Button variant="ghost" size="icon-xs" disabled aria-label="Reset brain camera">
            <RotateCcw aria-hidden="true" />
          </Button>
        </div>
      </TabsContent>
      <TabsContent value="details" className="inspector-prose">
        <span className="eyebrow">ABOUT THIS VIEW</span>
        <h2>A window into the network.</h2>
        <p>
          This image is an original, decorative point cloud. It is not a reconstruction of the
          connectome or a recorded neural response.
        </p>
        <p>
          Once connected, this view will show neurons at their real 3D coordinates, with brightness
          driven by streamed activity.
        </p>
        <dl className="detail-list">
          <div>
            <dt>Coordinates</dt>
            <dd>Awaiting backend</dd>
          </div>
          <div>
            <dt>Neuron count</dt>
            <dd>Not available</dd>
          </div>
          <div>
            <dt>Activity stream</dt>
            <dd>Not connected</dd>
          </div>
        </dl>
      </TabsContent>
      <TabsContent value="next" className="inspector-prose">
        <span className="eyebrow">IMPLEMENTATION ROADMAP</span>
        <h2>Two views. Full perspective.</h2>
        <ol className="next-steps">
          <li>
            <strong>3D Fly Saber</strong>
            <p>
              Live blocks and sabers, with an independent camera to orbit, pan, zoom, and reset.
            </p>
          </li>
          <li>
            <strong>A rotatable 3D brain</strong>
            <p>
              Real coordinates and activity, drag-to-orbit, camera presets, and neuron selection.
            </p>
          </li>
          <li>
            <strong>Live experiments</strong>
            <p>Lesion controls, real versus scrambled wiring, and a recorded replay fallback.</p>
          </li>
        </ol>
        <p className="plan-reference">
          The game and brain will share a simulation timeline, with independent camera controls for
          each view.
        </p>
      </TabsContent>
    </Tabs>
  )
}
