import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ComponentRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  NormalBlending,
  ShaderMaterial,
  LineBasicMaterial,
  Spherical,
  Vector3,
} from 'three'
import type { MutableRefObject } from 'react'
import { cellTypeColor } from './data'
import type { BrainData, BrainEdges } from './data'
import { liveActivity } from './live-activity'
import { CAMERA_SETTLE_MS, connectionIndex, MOVING_EDGE_LIMIT } from './rendering'
import { useScenePalette } from '@/lib/theme'

/*
 * Live activity is drawn as ink on paper, the light-plate counterpart of a glow on black.
 * Resting neurons are a faint pencil sketch of the anatomy. Active neurons are a second,
 * larger layer drawn over it: change from rest is signed, so excitation is vermilion and
 * inhibition blue (a diverging scale, as calcium imaging is printed). Normal blending keeps
 * overlapping actives in their own color; multiplying them turned red over blue to black.
 * Colors match --activity-* in index.css; selection uses amber and violet, never these.
 * In the dark theme the same scale is drawn brighter on black (see useScenePalette).
 */
// |change from rest| (of 127) that draws at full strength. The median active neuron moves
// about 3, so a linear scale would leave nearly every active neuron as faint as one at rest.
const FULL_AT = 32
// Seconds for displayed activity to close most of the gap to a new frame, so per-step
// flicker reads as pulses that rise and fade.
const EASE_SECONDS = 0.12

/** Signed display level in [-1, 1] for a change from rest. */
function activityLevel(change: number) {
  return Math.sign(change) * Math.min(1, Math.sqrt(Math.abs(change) / FULL_AT))
}

const VERTEX = /* glsl */ `
  attribute vec3 color;
  attribute float activity;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uLive;
  uniform float uInk;
  uniform vec3 uRest;
  uniform vec3 uExcite;
  uniform vec3 uInhibit;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    float level = abs(activity);
    if (uInk > 0.5) {
      // Active neurons only, from about 2 of 127 change: smaller drifts stay in the sketch.
      vColor = activity >= 0.0 ? uExcite : uInhibit;
      vAlpha = mix(0.45, 1.0, level);
      gl_PointSize = level < 0.25 ? 0.0 : uSize * uPixelRatio * mix(1.2, 2.6, level);
    } else if (uLive > 0.5) {
      vColor = uRest;
      vAlpha = 0.28;
      gl_PointSize = uSize * uPixelRatio * 0.8;
    } else {
      vColor = color;
      vAlpha = 1.0;
      gl_PointSize = uSize * uPixelRatio;
    }
  }
`
const FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  uniform float uInk;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float alpha = vAlpha * uOpacity * smoothstep(0.5, 0.3, d);
    gl_FragColor = vec4(vColor * alpha, alpha);
  }
`

/** Props for one layer's <shaderMaterial>; uniforms are updated in place each frame. */
function cloudMaterial(ink: boolean) {
  return {
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: {
      uSize: { value: 1.5 },
      uPixelRatio: { value: 1 },
      uOpacity: { value: 1 },
      uLive: { value: 0 },
      uInk: { value: ink ? 1 : 0 },
      uRest: { value: new Color() },
      uExcite: { value: new Color() },
      uInhibit: { value: new Color() },
    },
    transparent: true,
    premultipliedAlpha: true,
    blending: NormalBlending,
    depthWrite: false,
    depthTest: false,
  }
}

export type CameraAction =
  'reset' | 'x' | 'y' | 'z' | 'left' | 'right' | 'up' | 'down' | 'in' | 'out'

interface SceneProps {
  data: BrainData
  edges: BrainEdges | null
  indices: number[]
  edgeIndices: number[]
  showConnections: boolean
  selected: number | null
  onSelect: (index: number | null) => void
  onNavigateReady: (navigate: ((action: CameraAction) => void) | null) => void
  onFailure: () => void
  pointSize: number
  opacity: number
  onMotionChange: (moving: boolean) => void
}

function Connections({
  data,
  edges,
  edgeIndices,
  selected,
  moving,
}: Pick<SceneProps, 'data' | 'edgeIndices' | 'selected'> & {
  edges: BrainEdges
  moving: MutableRefObject<boolean>
}) {
  const palette = useScenePalette()
  const geometry = useMemo(() => {
    const { index, outgoing } = connectionIndex({ asset: data.asset, edges }, edgeIndices, selected)
    const result = new BufferGeometry()
    result.setAttribute('position', new BufferAttribute(data.positions, 3))
    result.setIndex(new BufferAttribute(index, 1))
    if (selected !== null) {
      result.addGroup(0, outgoing, 0)
      result.addGroup(outgoing, index.length - outgoing, 1)
    }
    result.computeBoundingSphere()
    return result
  }, [data, edges, edgeIndices, selected])
  useEffect(() => () => geometry.dispose(), [geometry])
  const quiet = useRef<LineBasicMaterial>(null)
  useFrame(() => {
    // While activity plays, the full web would bury the active neurons under gray.
    if (quiet.current) quiet.current.opacity = liveActivity.live() ? 0.01 : 0.022
    const count =
      moving.current && selected === null
        ? Math.min(edgeIndices.length, MOVING_EDGE_LIMIT)
        : edgeIndices.length
    geometry.setDrawRange(0, count * 2)
  })
  return (
    <lineSegments raycast={() => {}} renderOrder={-1}>
      <primitive object={geometry} attach="geometry" />
      {selected === null ? (
        <lineBasicMaterial
          ref={quiet}
          color={palette.quietEdge}
          transparent
          opacity={0.022}
          depthWrite={false}
          depthTest={false}
        />
      ) : (
        <>
          <lineBasicMaterial
            attach="material-0"
            color="#a86b00"
            transparent
            opacity={0.75}
            depthWrite={false}
            depthTest={false}
          />
          <lineBasicMaterial
            attach="material-1"
            color="#7847a6"
            transparent
            opacity={0.75}
            depthWrite={false}
            depthTest={false}
          />
        </>
      )}
    </lineSegments>
  )
}

function Navigation({
  data,
  onNavigateReady,
  onMotionChange,
}: Pick<SceneProps, 'data' | 'onNavigateReady' | 'onMotionChange'>) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)
  const fittedData = useRef<BrainData | null>(null)
  const moving = useRef(false)
  const { camera, size, invalidate } = useThree()
  const dragging = useRef(false)
  const lastChange = useRef(0)
  const settleTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const finishMotion = useCallback(() => {
    if (dragging.current || !moving.current) return
    moving.current = false
    clearTimeout(settleTimer.current)
    onMotionChange(false)
    // The demand loop may already be asleep; render the full graph once more.
    invalidate()
  }, [onMotionChange, invalidate])
  const queueSettle = useCallback(() => {
    lastChange.current = performance.now()
    clearTimeout(settleTimer.current)
    // Wake the demand loop after the final event. Finish inside useFrame, after
    // OrbitControls updates, so a slow frame cannot restore detail mid-damping.
    settleTimer.current = setTimeout(invalidate, CAMERA_SETTLE_MS + 1)
  }, [invalidate])
  const beginMotion = useCallback(() => {
    if (!moving.current) {
      moving.current = true
      onMotionChange(true)
    }
    queueSettle()
  }, [onMotionChange, queueSettle])
  const cameraChanged = useCallback(() => {
    // Tiny late damping changes can arrive after settling. Only a new input
    // gesture or navigation command may re-enter the reduced-detail preview.
    if (moving.current) queueSettle()
  }, [queueSettle])
  useFrame(() => {
    if (moving.current && performance.now() - lastChange.current >= CAMERA_SETTLE_MS) {
      finishMotion()
    }
  }, -0.5)
  useEffect(
    () => () => {
      clearTimeout(settleTimer.current)
      moving.current = false
      onMotionChange(false)
    },
    [moving, onMotionChange],
  )
  const aspect = size.width / Math.max(1, size.height)
  const distance =
    (data.radius /
      Math.sin(
        Math.min(Math.atan(Math.tan((21 * Math.PI) / 180) * aspect), (21 * Math.PI) / 180),
      )) *
    1.12
  const navigate = useCallback(
    (action: CameraAction) => {
      const orbit = controls.current
      if (!orbit) return
      beginMotion()
      if (['reset', 'x', 'y', 'z'].includes(action)) {
        orbit.target.set(0, 0, 0)
        camera.up.set(0, 1, 0)
        if (action === 'x') camera.position.set(distance, 0, 0)
        else if (action === 'y') {
          camera.position.set(0, distance, 0)
          camera.up.set(0, 0, -1)
        } else camera.position.set(0, 0, distance)
      } else {
        const offset = camera.position.clone().sub(orbit.target)
        if (action === 'in' || action === 'out') {
          offset.multiplyScalar(action === 'in' ? 0.82 : 1.22)
          offset.clampLength(data.radius * 0.3, distance * 4)
        } else {
          const spherical = new Spherical().setFromVector3(offset)
          spherical.theta += action === 'left' ? 0.18 : action === 'right' ? -0.18 : 0
          spherical.phi += action === 'up' ? -0.18 : action === 'down' ? 0.18 : 0
          spherical.makeSafe()
          offset.setFromSpherical(spherical)
        }
        camera.position.copy(orbit.target).add(offset)
      }
      camera.lookAt(orbit.target)
      orbit.update()
      invalidate()
    },
    [camera, data.radius, distance, invalidate, beginMotion],
  )

  useEffect(() => {
    // Resize updates the projection and future fit distance, never the user's
    // current orbit, zoom, or pan. Fit once for each newly loaded dataset.
    if (fittedData.current !== data) {
      fittedData.current = data
      navigate('reset')
    }
  }, [data, navigate])
  useEffect(() => {
    onNavigateReady(navigate)
    return () => onNavigateReady(null)
  }, [navigate, onNavigateReady])
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.18}
      onChange={cameraChanged}
      onStart={() => {
        dragging.current = true
        beginMotion()
      }}
      onEnd={() => {
        dragging.current = false
        cameraChanged()
      }}
      minDistance={data.radius * 0.3}
      maxDistance={distance * 4}
    />
  )
}

function NeuronCloud({
  data,
  indices,
  selected,
  onSelect,
  pointSize,
  opacity,
}: Pick<SceneProps, 'data' | 'indices' | 'selected' | 'onSelect' | 'pointSize' | 'opacity'>) {
  const palette = useScenePalette()
  const geometry = useMemo(() => {
    const positions = new Float32Array(indices.length * 3)
    const colors = new Float32Array(indices.length * 3)
    const palette = data.asset.cell_types.map((_, type) => new Color(cellTypeColor(type)))
    indices.forEach((index, offset) => {
      positions.set(data.positions.subarray(index * 3, index * 3 + 3), offset * 3)
      palette[data.asset.type[index]].toArray(colors, offset * 3)
    })
    const value = new BufferGeometry()
    value.setAttribute('position', new BufferAttribute(positions, 3))
    value.setAttribute('color', new BufferAttribute(colors, 3))
    value.setAttribute('activity', new BufferAttribute(new Float32Array(indices.length), 1))
    return value
  }, [data, indices])
  useEffect(() => () => geometry.dispose(), [geometry])
  const sketchProps = useMemo(() => cloudMaterial(false), [])
  const inkProps = useMemo(() => cloudMaterial(true), [])
  const sketchRef = useRef<ShaderMaterial>(null)
  const inkRef = useRef<ShaderMaterial>(null)
  // Target levels for the latest activity step; displayed levels ease toward them.
  const target = useRef({ geometry, step: -1, values: new Float32Array(indices.length) })
  const live = useRef(false)
  const invalidate = useThree((state) => state.invalidate)
  const pixelRatio = useThree((state) => state.viewport.dpr)
  useEffect(() => liveActivity.subscribe(invalidate), [invalidate])
  useEffect(() => invalidate(), [palette, invalidate])
  useFrame((_, delta) => {
    const sketch = sketchRef.current
    const ink = inkRef.current
    if (!sketch || !ink) return
    for (const material of [sketch, ink]) {
      material.uniforms.uRest.value.set(palette.restInk)
      material.uniforms.uExcite.value.set(palette.excite)
      material.uniforms.uInhibit.value.set(palette.inhibit)
      material.uniforms.uSize.value = pointSize
      material.uniforms.uOpacity.value = opacity
      material.uniforms.uPixelRatio.value = pixelRatio
    }
    const neurons = liveActivity.neurons
    const isLive = liveActivity.live() && neurons?.length === data.asset.n_neurons
    const goal = target.current
    if (goal.geometry !== geometry) {
      target.current = { geometry, step: -1, values: new Float32Array(indices.length) }
      return invalidate()
    }
    if (isLive && neurons && goal.step !== liveActivity.neuronStep) {
      indices.forEach((index, offset) => (goal.values[offset] = activityLevel(neurons[index])))
      goal.step = liveActivity.neuronStep
    } else if (!isLive && goal.step !== -1) {
      goal.values.fill(0)
      goal.step = -1
    }
    if (live.current !== isLive) {
      live.current = isLive
      sketch.uniforms.uLive.value = isLive ? 1 : 0
    }
    // Ease displayed activity toward the target; stop redrawing once it has settled.
    const attribute = geometry.getAttribute('activity') as BufferAttribute
    const shown = attribute.array as Float32Array
    const step = 1 - Math.exp(-Math.min(delta, 0.1) / EASE_SECONDS)
    let moving = false
    for (let i = 0; i < shown.length; i++) {
      const gap = goal.values[i] - shown[i]
      if (gap === 0) continue
      if (Math.abs(gap) < 0.004) shown[i] = goal.values[i]
      else {
        shown[i] += gap * step
        moving = true
      }
      attribute.needsUpdate = true
    }
    ink.visible = isLive || moving
    if (moving) invalidate()
  })
  const selectedPosition =
    selected === null ? null : new Vector3().fromArray(data.positions, selected * 3)
  return (
    <>
      <points
        geometry={geometry}
        onClick={(event) => {
          if (event.delta > 4 || event.index === undefined) return
          event.stopPropagation()
          onSelect(indices[event.index])
        }}
      >
        <shaderMaterial ref={sketchRef} {...sketchProps} />
      </points>
      <points geometry={geometry} raycast={() => {}} renderOrder={1}>
        <shaderMaterial ref={inkRef} {...inkProps} />
      </points>
      {selectedPosition && (
        <mesh position={selectedPosition} raycast={() => {}}>
          <sphereGeometry args={[0.03, 12, 8]} />
          <meshBasicMaterial color={palette.selection} wireframe depthTest={false} />
        </mesh>
      )}
    </>
  )
}

export default function BrainScene(props: SceneProps) {
  const palette = useScenePalette()
  const moving = useRef(false)
  const [cameraMoving, setCameraMoving] = useState(false)
  const { onMotionChange } = props
  const updateMotion = useCallback(
    (value: boolean) => {
      moving.current = value
      setCameraMoving(value)
      onMotionChange(value)
    },
    [onMotionChange],
  )
  return (
    <Canvas
      frameloop="demand"
      dpr={cameraMoving ? 1 : [1, 1.75]}
      camera={{ fov: 42, near: 0.01, far: 100 }}
      gl={{ antialias: true, alpha: true }}
      raycaster={{
        params: {
          Mesh: {},
          Line: { threshold: 1 },
          LOD: {},
          Sprite: {},
          Points: { threshold: 0.018 },
        },
      }}
      onCreated={({ gl }) => {
        gl.domElement.setAttribute('aria-label', '3D neurons and directed connections')
      }}
      onContextMenu={(event) => event.preventDefault()}
      onPointerMissed={(event) => {
        if (event.type === 'click') props.onSelect(null)
      }}
    >
      <color attach="background" args={[palette.field]} />
      <ContextMonitor onFailure={props.onFailure} />
      <Navigation
        data={props.data}
        onNavigateReady={props.onNavigateReady}
        onMotionChange={updateMotion}
      />
      {props.showConnections && props.edges && (
        <Connections {...props} edges={props.edges} moving={moving} />
      )}
      <NeuronCloud {...props} />
    </Canvas>
  )
}

function ContextMonitor({ onFailure }: Pick<SceneProps, 'onFailure'>) {
  const canvas = useThree((state) => state.gl.domElement)
  useEffect(() => {
    const lost = (event: Event) => {
      event.preventDefault()
      onFailure()
    }
    canvas.addEventListener('webglcontextlost', lost)
    return () => canvas.removeEventListener('webglcontextlost', lost)
  }, [canvas, onFailure])
  return null
}
