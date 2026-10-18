import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ComponentRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { BufferAttribute, BufferGeometry, Color, Spherical, Vector3 } from 'three'
import type { MutableRefObject } from 'react'
import { cellTypeColor } from './data'
import type { BrainData, BrainEdges } from './data'
import { liveActivity } from './live-activity'
import { CAMERA_SETTLE_MS, connectionIndex, MOVING_EDGE_LIMIT } from './rendering'

// Brightness of a neuron at rest while a brain plays, so active neurons stand out.
const REST_BRIGHTNESS = 0.15
// |change from rest| (of 127) that draws at full brightness. The median active neuron moves
// about 3, so a linear scale would leave nearly every active neuron as dark as one at rest.
const FULL_BRIGHTNESS_AT = 32

function brightness(change: number) {
  const level = Math.min(1, Math.sqrt(Math.abs(change) / FULL_BRIGHTNESS_AT))
  return REST_BRIGHTNESS + (1 - REST_BRIGHTNESS) * level
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
  useFrame(() => {
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
          color="#3f4a4b"
          transparent
          opacity={0.035}
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
            color="#2b63b0"
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
  const buffers = useMemo(() => {
    const positions = new Float32Array(indices.length * 3)
    const colors = new Float32Array(indices.length * 3)
    const palette = data.asset.cell_types.map((_, type) => new Color(cellTypeColor(type)))
    indices.forEach((index, offset) => {
      positions.set(data.positions.subarray(index * 3, index * 3 + 3), offset * 3)
      palette[data.asset.type[index]].toArray(colors, offset * 3)
    })
    return { positions, colors, base: colors.slice() }
  }, [data, indices])
  const colorAttribute = useRef<BufferAttribute>(null)
  const drawn = useRef({ buffers, step: -1, live: false })
  const invalidate = useThree((state) => state.invalidate)
  useEffect(() => liveActivity.subscribe(invalidate), [invalidate])
  useFrame(() => {
    const neurons = liveActivity.neurons
    const live = liveActivity.live() && neurons?.length === data.asset.n_neurons
    const last = drawn.current
    if (
      last.buffers === buffers &&
      last.live === live &&
      (!live || last.step === liveActivity.neuronStep)
    )
      return
    const { colors, base } = buffers
    if (live && neurons) {
      indices.forEach((index, offset) => {
        const level = brightness(neurons[index])
        for (let i = offset * 3; i < offset * 3 + 3; i++) colors[i] = base[i] * level
      })
    } else colors.set(base)
    if (colorAttribute.current) colorAttribute.current.needsUpdate = true
    drawn.current = { buffers, step: liveActivity.neuronStep, live }
  })
  const selectedPosition =
    selected === null ? null : new Vector3().fromArray(data.positions, selected * 3)
  return (
    <>
      <points
        onClick={(event) => {
          if (event.delta > 4 || event.index === undefined) return
          event.stopPropagation()
          onSelect(indices[event.index])
        }}
      >
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[buffers.positions, 3]} />
          <bufferAttribute
            ref={colorAttribute}
            attach="attributes-color"
            args={[buffers.colors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={pointSize}
          vertexColors
          sizeAttenuation={false}
          transparent
          opacity={opacity}
          depthWrite={false}
          depthTest={false}
        />
      </points>
      {selectedPosition && (
        <mesh position={selectedPosition} raycast={() => {}}>
          <sphereGeometry args={[0.03, 12, 8]} />
          <meshBasicMaterial color="#162021" wireframe depthTest={false} />
        </mesh>
      )}
    </>
  )
}

export default function BrainScene(props: SceneProps) {
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
