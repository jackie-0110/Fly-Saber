import { useEffect, useMemo, useRef } from 'react'
import type { RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Quaternion, Vector3 } from 'three'
import type { Group, Mesh } from 'three'
import { FlyMotionController } from './fly-motion'

const COLORS = ['#d861aa', '#5ac576']

function Segment({
  from,
  to,
  color = '#5d696a',
  radius = 0.035,
}: {
  from: [number, number, number]
  to: [number, number, number]
  color?: string
  radius?: number
}) {
  const { middle, rotation, length } = useMemo(() => {
    const a = new Vector3(...from)
    const b = new Vector3(...to)
    const direction = b.clone().sub(a)
    return {
      middle: a.add(b).multiplyScalar(0.5),
      length: direction.length(),
      rotation: new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), direction.normalize()),
    }
  }, [from, to])
  return (
    <mesh position={middle} quaternion={rotation}>
      <cylinderGeometry args={[radius, radius, length, 6]} />
      <meshStandardMaterial color={color} roughness={0.6} />
    </mesh>
  )
}

function StaticFlyBody({ wings }: { wings: RefObject<(Group | null)[]> }) {
  return (
    <group>
      <mesh position={[0, 0.98, 0.48]} scale={[0.54, 0.55, 0.86]}>
        <sphereGeometry args={[0.65, 16, 12]} />
        <meshStandardMaterial color="#595e56" roughness={0.7} metalness={0.15} />
      </mesh>
      <mesh position={[0, 1.16, -0.18]} scale={[0.8, 0.85, 1]}>
        <sphereGeometry args={[0.42, 16, 12]} />
        <meshStandardMaterial color="#7c8170" roughness={0.75} />
      </mesh>
      <mesh position={[0, 1.44, -0.62]} scale={[1, 0.85, 0.75]}>
        <sphereGeometry args={[0.37, 16, 12]} />
        <meshStandardMaterial color="#acac8d" roughness={0.7} />
      </mesh>
      {[-1, 1].map((side) => (
        <group key={side}>
          <mesh position={[side * 0.28, 1.5, -0.72]} scale={[0.6, 0.95, 0.7]}>
            <sphereGeometry args={[0.27, 12, 10]} />
            <meshStandardMaterial color="#b66855" roughness={0.45} />
          </mesh>
          <group
            ref={(node) => {
              wings.current[side < 0 ? 0 : 1] = node
            }}
            position={[side * 0.2, 1.43, 0.36]}
          >
            <mesh
              position={[side * 0.62, 0, 0]}
              rotation={[0.1, side * 0.35, 0]}
              scale={[1.24, 0.035, 0.56]}
            >
              <sphereGeometry args={[0.8, 16, 8]} />
              <meshStandardMaterial
                color="#b8d5ce"
                transparent
                opacity={0.28}
                roughness={0.25}
                depthWrite={false}
              />
            </mesh>
          </group>
          <Segment
            from={[side * 0.15, 1.67, -0.73]}
            to={[side * 0.3, 1.98, -1.02]}
            radius={0.018}
          />
          {[1, 2].map((leg) => (
            <group key={leg}>
              <Segment
                from={[side * 0.2, 1, leg * 0.3 - 0.2]}
                to={[side * 0.7, 0.5, leg * 0.55 - 0.45]}
              />
              <Segment
                from={[side * 0.7, 0.5, leg * 0.55 - 0.45]}
                to={[side * 0.95, 0.03, leg * 0.65 - 0.3]}
                radius={0.025}
              />
            </group>
          ))}
        </group>
      ))}
    </group>
  )
}

function Limb({ limbRef }: { limbRef: (mesh: Mesh | null) => void }) {
  return (
    <mesh ref={limbRef}>
      <cylinderGeometry args={[0.035, 0.045, 1, 8]} />
      <meshStandardMaterial color="#737a6a" roughness={0.55} />
    </mesh>
  )
}

/** This component is the mesh rig; the controller does not depend on its geometry. */
export function FlyModel({ controller }: { controller: FlyMotionController }) {
  const body = useRef<Group>(null)
  const wings = useRef<(Group | null)[]>([])
  const hands = useRef<(Group | null)[]>([])
  const bones = useRef<(Mesh | null)[]>([])
  const { invalidate } = useThree()
  const reduced = useRef(false)
  const scratch = useMemo(
    () => ({
      shoulder: new Vector3(),
      grip: new Vector3(),
      elbow: new Vector3(),
      direction: new Vector3(),
      bend: new Vector3(),
      up: new Vector3(0, 1, 0),
    }),
    [],
  )

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)')
    const change = () => {
      reduced.current = preference.matches
      invalidate()
    }
    change()
    preference.addEventListener('change', change)
    const unsubscribe = controller.subscribe(invalidate)
    // Only this mounted, visible rig requests ambient frames; hidden tabs suspend rAF.
    let handle = 0
    let last = -Infinity
    const tick = (milliseconds: number) => {
      const hz = controller.sample(milliseconds / 1000).active ? 60 : 20
      if (milliseconds - last >= 1000 / hz) {
        last = milliseconds
        invalidate()
      }
      handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(handle)
      unsubscribe()
      preference.removeEventListener('change', change)
    }
  }, [controller, invalidate])

  useFrame(() => {
    if (!body.current) return
    const pose = controller.sample()
    const energy = Math.max(...pose.hands.map((hand) => hand.energy))
    const sway = reduced.current ? 0 : Math.sin(pose.time * 2.3) * 0.018
    const lean = (pose.hands[0].center[0] + pose.hands[1].center[0]) * 0.045
    body.current.position.set(lean, 0.05 + sway + energy * 0.035, 2.6 - energy * 0.09)
    body.current.rotation.set(energy * 0.04, -lean * 0.3, -lean * 0.18)
    body.current.updateMatrixWorld(true)
    for (const color of [0, 1] as const) {
      const side = color === 0 ? -1 : 1
      const hand = pose.hands[color]
      const saber = hands.current[color]
      if (saber) {
        // Blade midpoint is the grid target; the foreleg grips its hilt below it.
        scratch.grip.set(
          hand.center[0] + Math.sin(hand.angle) * 0.88,
          hand.center[1] - Math.cos(hand.angle) * 0.88,
          hand.center[2],
        )
        saber.position.copy(scratch.grip)
        saber.rotation.z = hand.angle
      }
      if (wings.current[color])
        wings.current[color]!.rotation.z =
          side *
          (0.17 +
            (reduced.current
              ? 0
              : Math.sin(pose.time * (10 + energy * 12)) * (0.025 + energy * 0.08)))
      scratch.shoulder.set(side * 0.23, 1.12, -0.25)
      body.current.localToWorld(scratch.shoulder)
      scratch.direction.copy(scratch.grip).sub(scratch.shoulder)
      // A two-segment foreleg bends outwards; lengths stretch for the stylized grid reach.
      scratch.bend.set(side, -0.4, 0)
      const length = scratch.direction.length()
      scratch.direction.normalize()
      scratch.bend
        .addScaledVector(scratch.direction, -scratch.bend.dot(scratch.direction))
        .normalize()
      scratch.elbow
        .copy(scratch.shoulder)
        .lerp(scratch.grip, 0.5)
        .addScaledVector(scratch.bend, Math.max(0.22, 0.7 - length * 0.1))
      for (let part = 0; part < 2; part++) {
        const bone = bones.current[color * 2 + part]
        if (!bone) continue
        const from = part === 0 ? scratch.shoulder : scratch.elbow
        const to = part === 0 ? scratch.elbow : scratch.grip
        scratch.direction.copy(to).sub(from)
        bone.position.copy(from).lerp(to, 0.5)
        bone.scale.y = scratch.direction.length()
        bone.quaternion.setFromUnitVectors(scratch.up, scratch.direction.normalize())
      }
    }
  })

  return (
    <group name="fly-motion-rig">
      <group ref={body} name="fly-body">
        <StaticFlyBody wings={wings} />
      </group>
      {[0, 1].map((color) => (
        <group key={color}>
          {[0, 1].map((part) => (
            <Limb
              key={part}
              limbRef={(mesh) => {
                bones.current[color * 2 + part] = mesh
              }}
            />
          ))}
          <group
            name={color === 0 ? 'left-saber' : 'right-saber'}
            ref={(node) => {
              hands.current[color] = node
            }}
          >
            <mesh>
              <cylinderGeometry args={[0.055, 0.075, 0.35, 10]} />
              <meshStandardMaterial color="#737a7c" metalness={0.7} roughness={0.35} />
            </mesh>
            <mesh position={[0, 0.88, 0]}>
              <cylinderGeometry args={[0.025, 0.035, 1.42, 10]} />
              <meshBasicMaterial color={COLORS[color]} />
            </mesh>
            <mesh position={[0, 0.88, 0]}>
              <cylinderGeometry args={[0.065, 0.065, 1.4, 10]} />
              <meshBasicMaterial
                color={COLORS[color]}
                transparent
                opacity={0.16}
                depthWrite={false}
              />
            </mesh>
          </group>
        </group>
      ))}
    </group>
  )
}
