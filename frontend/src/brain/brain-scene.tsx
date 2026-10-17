import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { ComponentRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Color, Spherical, Vector3 } from 'three'
import { cellTypeColor } from './data'
import type { BrainData } from './data'

export type CameraAction =
  'reset' | 'x' | 'y' | 'z' | 'left' | 'right' | 'up' | 'down' | 'in' | 'out'

interface SceneProps {
  data: BrainData
  indices: number[]
  selected: number | null
  onSelect: (index: number | null) => void
  onNavigateReady: (navigate: ((action: CameraAction) => void) | null) => void
  onFailure: () => void
  pointSize: number
  opacity: number
}

function Navigation({ data, onNavigateReady }: Pick<SceneProps, 'data' | 'onNavigateReady'>) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)
  const { camera, size, invalidate } = useThree()
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
    [camera, data.radius, distance, invalidate],
  )

  useEffect(() => {
    navigate('reset')
  }, [navigate])
  useEffect(() => {
    onNavigateReady(navigate)
    return () => onNavigateReady(null)
  }, [navigate, onNavigateReady])
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
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
    return { positions, colors }
  }, [data, indices])
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
          <bufferAttribute attach="attributes-color" args={[buffers.colors, 3]} />
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
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.75]}
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
        gl.domElement.setAttribute('aria-label', '3D neuron point cloud')
      }}
      onContextMenu={(event) => event.preventDefault()}
      onPointerMissed={(event) => {
        if (event.type === 'click') props.onSelect(null)
      }}
    >
      <ContextMonitor onFailure={props.onFailure} />
      <Navigation data={props.data} onNavigateReady={props.onNavigateReady} />
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
