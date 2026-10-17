import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { ComponentRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import {
  BoxGeometry,
  CanvasTexture,
  IcosahedronGeometry,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  Spherical,
  SRGBColorSpace,
} from 'three'
import type { InstancedMesh, Material } from 'three'
import { lowerBound } from './chart'
import type { Chart } from './chart'
import { transport, usePlayer } from './store'
import { FlyModel } from './fly-model'
import { flyMotion, previewMotion } from './fly-motion'
import { useSession } from './session'

export type GameCameraAction =
  'reset' | 'overview' | 'player' | 'left' | 'right' | 'up' | 'down' | 'in' | 'out'
interface SceneProps {
  chart: Chart | null
  showFly: boolean
  motionPreview?: boolean
  onNavigateReady: (navigate: ((action: GameCameraAction) => void) | null) => void
  onFailure: () => void
}

const CAPACITY = 1024
const WALL_CAPACITY = 128
const COLORS = ['#e6a08b', '#a4d9cf']

function faceTexture(color: string, cut: number) {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 256
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = color
  ctx.fillRect(0, 0, 256, 256)
  if (cut !== 8) {
    ctx.save()
    ctx.translate(128, 128)
    ctx.rotate(
      [
        0,
        Math.PI,
        -Math.PI / 2,
        Math.PI / 2,
        -Math.PI / 4,
        Math.PI / 4,
        (-3 * Math.PI) / 4,
        (3 * Math.PI) / 4,
      ][cut],
    )
    ctx.fillStyle = '#00000065'
    ctx.fillRect(-190, -190, 380, 190)
    ctx.fillStyle = '#172125'
    ctx.beginPath()
    ctx.moveTo(0, -72)
    ctx.lineTo(61, -13)
    ctx.lineTo(23, -13)
    ctx.lineTo(23, 65)
    ctx.lineTo(-23, 65)
    ctx.lineTo(-23, -13)
    ctx.lineTo(-61, -13)
    ctx.closePath()
    ctx.fill()
    ctx.restore()
  } else {
    ctx.beginPath()
    ctx.arc(128, 128, 26, 0, Math.PI * 2)
    ctx.fillStyle = '#172125'
    ctx.fill()
  }
  ctx.strokeStyle = '#ffffff30'
  ctx.lineWidth = 8
  ctx.strokeRect(4, 4, 248, 248)
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  texture.anisotropy = 4
  return texture
}

function Navigation({ onNavigateReady }: Pick<SceneProps, 'onNavigateReady'>) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)
  const { camera, invalidate, size } = useThree()
  const narrow = size.width / Math.max(size.height, 1) < 1
  const navigate = useCallback(
    (action: GameCameraAction) => {
      const orbit = controls.current
      if (!orbit) return
      if (action === 'reset' || action === 'overview' || action === 'player') {
        const positions = {
          reset: narrow ? [10, 8, 17] : [8.5, 6.5, 13],
          overview: [12, 19, 8],
          player: [0, 2.5, 7.5],
        }
        camera.position.fromArray(positions[action])
        orbit.target.set(0, action === 'player' ? 1.5 : 0.8, action === 'overview' ? -12 : -5)
      } else {
        const offset = camera.position.clone().sub(orbit.target)
        if (action === 'in' || action === 'out')
          offset.multiplyScalar(action === 'in' ? 0.82 : 1.22).clampLength(4, 70)
        else {
          const spherical = new Spherical().setFromVector3(offset)
          spherical.theta += action === 'left' ? 0.15 : action === 'right' ? -0.15 : 0
          spherical.phi += action === 'up' ? -0.12 : action === 'down' ? 0.12 : 0
          spherical.phi = Math.max(0.05, Math.min(Math.PI * 0.49, spherical.phi))
          offset.setFromSpherical(spherical)
        }
        camera.position.copy(orbit.target).add(offset)
      }
      camera.lookAt(orbit.target)
      orbit.update()
      invalidate()
    },
    [camera, invalidate, narrow],
  )

  // Set the initial view once; viewport resizing and song changes preserve the chosen camera.
  const initialized = useRef(false)
  useEffect(() => {
    if (!initialized.current) {
      navigate('reset')
      initialized.current = true
    }
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
      dampingFactor={0.09}
      minDistance={4}
      maxDistance={70}
      maxPolarAngle={Math.PI * 0.49}
      minPolarAngle={0.05}
      zoomSpeed={0.8}
      panSpeed={0.65}
    />
  )
}

function Track() {
  const lines = useMemo(() => {
    const vertices: number[] = []
    for (let z = -80; z <= 8; z += 4) vertices.push(-3.2, -0.18, z, 3.2, -0.18, z)
    for (const x of [-2.4, -1.2, 0, 1.2, 2.4]) vertices.push(x, -0.18, 8, x, -0.18, -80)
    return new Float32Array(vertices)
  }, [])
  return (
    <group>
      <mesh position={[0, -0.31, -32]}>
        <boxGeometry args={[7, 0.22, 88]} />
        <meshStandardMaterial color="#242c31" roughness={0.85} metalness={0.3} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[lines, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#536367" transparent opacity={0.32} />
      </lineSegments>
      {[-1, 1].map((side) => (
        <group key={side}>
          <mesh position={[side * 3.4, -0.12, -32]}>
            <boxGeometry args={[0.12, 0.18, 88]} />
            <meshStandardMaterial color="#536463" metalness={0.5} roughness={0.5} />
          </mesh>
          <mesh position={[side * 3.4, -0.015, -32]}>
            <boxGeometry args={[0.035, 0.015, 88]} />
            <meshBasicMaterial color={side < 0 ? '#b2796b' : '#7aaba3'} />
          </mesh>
          {[-12, -30, -48, -66].map((z) => (
            <group key={z} position={[side * 5.2, 0, z]}>
              <mesh position={[0, 3.2, 0]}>
                <boxGeometry args={[0.35, 7, 0.35]} />
                <meshStandardMaterial color="#283339" roughness={0.7} metalness={0.5} />
              </mesh>
              <mesh position={[side * -0.19, 3.2, 0.19]}>
                <boxGeometry args={[0.04, 4.7, 0.015]} />
                <meshBasicMaterial color={side < 0 ? '#785b54' : '#4c7470'} />
              </mesh>
            </group>
          ))}
        </group>
      ))}
      <mesh position={[0, -0.16, 0]}>
        <boxGeometry args={[6.5, 0.03, 0.08]} />
        <meshBasicMaterial color="#c3d7cf" />
      </mesh>
      <mesh position={[0, -0.42, -20]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[400, 400]} />
        <meshStandardMaterial color="#141a1e" roughness={1} />
      </mesh>
    </group>
  )
}

function Notes({ chart }: { chart: Chart | null }) {
  const instances = useRef<(InstancedMesh | null)[]>([])
  const wallMesh = useRef<InstancedMesh>(null)
  const resources = useMemo(() => {
    const cube = new BoxGeometry(0.96, 0.96, 0.96)
    const bomb = new IcosahedronGeometry(0.48, 0)
    const wall = new BoxGeometry(1, 1, 1)
    const textures: CanvasTexture[] = []
    const materials: Material[] = []
    const faces = COLORS.flatMap((color) => {
      const side = new MeshStandardMaterial({ color, metalness: 0.15, roughness: 0.5 })
      materials.push(side)
      return Array.from({ length: 9 }, (_, cut) => {
        const texture = faceTexture(color, cut)
        const front = new MeshBasicMaterial({ map: texture })
        textures.push(texture)
        materials.push(front)
        return [side, side, side, side, front, side]
      })
    })
    const bombMaterial = new MeshStandardMaterial({
      color: '#776772',
      metalness: 0.5,
      roughness: 0.5,
    })
    const wallMaterial = new MeshBasicMaterial({
      color: '#a59a86',
      wireframe: true,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    })
    materials.push(bombMaterial, wallMaterial)
    return {
      cube,
      bomb,
      wall,
      faces,
      bombMaterial,
      wallMaterial,
      textures,
      materials,
      dummy: new Object3D(),
    }
  }, [])
  useEffect(
    () => () => {
      resources.cube.dispose()
      resources.bomb.dispose()
      resources.wall.dispose()
      resources.textures.forEach((t) => t.dispose())
      resources.materials.forEach((m) => m.dispose())
    },
    [resources],
  )
  const maximumWallDuration = useMemo(
    () => Math.max(0, ...(chart?.walls.map((w) => w.duration_sec) ?? [])),
    [chart],
  )
  useFrame(() => {
    const counts = Array<number>(19).fill(0)
    const t = transport.getTime()
    if (chart) {
      const speed = chart.difficulty.note_speed_mps
      const spawn = chart.difficulty.spawn_distance_m
      const first = lowerBound(chart.notes, t - 0.9 / speed)
      const end = lowerBound(chart.notes, t + spawn / speed)
      const dummy = resources.dummy
      dummy.rotation.set(0, 0, 0)
      dummy.scale.set(1, 1, 1)
      for (let i = first; i < end; i++) {
        const note = chart.notes[i]
        if (useSession.getState().removed.has(note.id)) continue
        const category = note.is_bomb ? 18 : note.color! * 9 + note.cut_direction!
        if (counts[category] >= CAPACITY) continue
        dummy.position.set(
          (note.lane - 1.5) * 1.2,
          note.layer * 1.1 + 0.45,
          -(note.time_sec - t) * speed,
        )
        dummy.updateMatrix()
        instances.current[category]?.setMatrixAt(counts[category]++, dummy.matrix)
      }
      let walls = 0
      const firstWall = lowerBound(chart.walls, t - maximumWallDuration - 1 / speed)
      for (let i = firstWall; i < chart.walls.length && walls < WALL_CAPACITY; i++) {
        const wall = chart.walls[i]
        const front = (wall.time_sec - t) * speed
        if (front > spawn) break
        const back = (wall.time_sec + wall.duration_sec - t) * speed
        if (back < -1) continue
        const near = Math.max(-1, front)
        const far = Math.min(spawn, back)
        const height = Math.min(6, wall.height * 0.8)
        dummy.position.set(
          (wall.lane + wall.width / 2 - 2) * 1.2,
          wall.layer * 1.1 + height / 2 - 0.15,
          -(near + far) / 2,
        )
        dummy.scale.set(wall.width * 1.2, height, Math.max(0.05, far - near))
        dummy.updateMatrix()
        wallMesh.current?.setMatrixAt(walls++, dummy.matrix)
      }
      if (wallMesh.current) {
        wallMesh.current.count = walls
        wallMesh.current.instanceMatrix.needsUpdate = true
      }
    } else if (wallMesh.current) wallMesh.current.count = 0
    instances.current.forEach((mesh, category) => {
      if (mesh) {
        mesh.count = counts[category]
        mesh.instanceMatrix.needsUpdate = true
      }
    })
  })
  return (
    <group dispose={null}>
      {resources.faces.map((material, index) => (
        <instancedMesh
          key={index}
          ref={(node) => {
            instances.current[index] = node
          }}
          args={[resources.cube, material, CAPACITY]}
          count={0}
          frustumCulled={false}
        />
      ))}
      <instancedMesh
        ref={(node) => {
          instances.current[18] = node
        }}
        args={[resources.bomb, resources.bombMaterial, CAPACITY]}
        count={0}
        frustumCulled={false}
      />
      <instancedMesh
        ref={wallMesh}
        args={[resources.wall, resources.wallMaterial, WALL_CAPACITY]}
        count={0}
        frustumCulled={false}
      />
    </group>
  )
}

function SceneLifecycle({ onFailure }: Pick<SceneProps, 'onFailure'>) {
  const { gl, invalidate } = useThree()
  useEffect(() => {
    const canvas = gl.domElement
    const lost = (event: Event) => {
      event.preventDefault()
      onFailure()
    }
    canvas.addEventListener('webglcontextlost', lost)
    const unsubscribe = usePlayer.subscribe((state, previous) => {
      if (
        state.time !== previous.time ||
        state.chart !== previous.chart ||
        state.playing !== previous.playing
      )
        invalidate()
    })
    return () => {
      canvas.removeEventListener('webglcontextlost', lost)
      unsubscribe()
    }
  }, [gl, invalidate, onFailure])
  return null
}

export default function GameScene(props: SceneProps) {
  const playing = usePlayer((s) => s.playing)
  return (
    <Canvas
      frameloop={playing ? 'always' : 'demand'}
      dpr={[1, 1.5]}
      camera={{ fov: 46, near: 0.1, far: 250, position: [8.5, 6.5, 13] }}
      gl={{ antialias: true, alpha: false }}
      onCreated={({ gl }) => {
        gl.domElement.setAttribute('aria-label', '3D Fly Saber track')
      }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <color attach="background" args={['#171d22']} />
      <fog attach="fog" args={['#171d22', 25, 100]} />
      <ambientLight intensity={0.9} />
      <hemisphereLight args={['#c2d8d4', '#1b2429', 1.6]} />
      <directionalLight position={[5, 12, 10]} intensity={2.3} color="#ffe7d5" />
      <directionalLight position={[-8, 5, -10]} intensity={1.2} color="#9cc8cb" />
      <SceneLifecycle onFailure={props.onFailure} />
      <Navigation onNavigateReady={props.onNavigateReady} />
      <Track />
      <Notes chart={props.chart} />
      {props.showFly && <FlyModel controller={props.motionPreview ? previewMotion : flyMotion} />}
    </Canvas>
  )
}
