/** Visual motor adapter for docs/PROTOCOL.md §2. No map lookahead or scoring. */
export type SaberColor = 0 | 1
export type CutDirection = 0 | 1 | 2 | 3
export type Point3 = [number, number, number]
export interface SaberAction {
  color: SaberColor
  cell: number | null
  cut: CutDirection | null
}
export interface MotorFrame {
  type: 'frame'
  slot: 0 | 1
  render_state: { t: number; step: number; sabers: SaberAction[] }
}
export interface HandPose {
  center: Point3
  angle: number
  energy: number
}
export interface FlyPose {
  hands: [HandPose, HandPose]
  time: number
  active: boolean
}
interface HandTrack {
  action: SaberAction
  from: HandPose
  changed: number
  strike: number
  pending?: SaberAction
}

export const CUT_VECTORS: readonly [number, number][] = [
  [0, 1],
  [0, -1],
  [-1, 0],
  [1, 0],
]
export const STROKE_SECONDS = 0.14
export const STREAM_TIMEOUT = 0.35
const RETURN_SECONDS = 0.25
const smooth = (v: number) => {
  const x = Math.max(0, Math.min(1, v))
  return x * x * (3 - 2 * x)
}
const mix = (a: number, b: number, t: number) => a + (b - a) * t
const idle = (color: SaberColor): HandPose => ({
  center: [color === 0 ? -1.08 : 1.08, 1.8, 1.45],
  angle: color === 0 ? 0.4 : -0.4,
  energy: 0,
})
export function cellCenter(cell: number): Point3 {
  return [((cell % 4) - 1.5) * 1.2, Math.floor(cell / 4) * 1.1 + 0.45, 0]
}
function blend(a: HandPose, b: HandPose, amount: number): HandPose {
  // Take the shortest wrist rotation, including across ±π.
  const turn = Math.atan2(Math.sin(b.angle - a.angle), Math.cos(b.angle - a.angle))
  return {
    center: a.center.map((v, i) => mix(v, b.center[i], amount)) as Point3,
    angle: a.angle + turn * amount,
    energy: mix(a.energy, b.energy, amount),
  }
}
function poseForTrack(track: HandTrack, t: number): HandPose {
  const end = track.strike + STROKE_SECONDS
  if (track.pending && t >= end) {
    return poseForTrack(
      {
        action: track.pending,
        from: poseForTrack({ ...track, pending: undefined }, end),
        changed: end,
        strike: end,
      },
      t,
    )
  }
  const rest = idle(track.action.color)
  let target = rest
  if (track.action.cell !== null) {
    const center = cellCenter(track.action.cell)
    target = { center, angle: rest.angle * 0.5, energy: 0.25 }
    if (track.action.cut !== null) {
      const progress = Math.max(0, Math.min(1, (t - track.strike) / STROKE_SECONDS))
      const recover = 1 - smooth((t - track.strike - STROKE_SECONDS) / RETURN_SECONDS)
      const [dx, dy] = CUT_VECTORS[track.action.cut]
      const travel = (smooth(progress) - 0.5) * 1.3 * recover
      target = {
        center: [center[0] + dx * travel, center[1] + dy * travel, 0],
        angle: Math.atan2(-dx, dy) + Math.PI / 2 + (progress - 0.5) * 1.5 * recover,
        energy: Math.sin(progress * Math.PI) * recover,
      }
    }
  }
  const pose = blend(
    track.from,
    target,
    smooth((t - track.changed) / (track.action.cell === null ? RETURN_SECONDS : 0.035)),
  )

  // Keep low-cell grips and blades above the track without changing the grid target.
  pose.center[1] = Math.max(0.12, pose.center[1])
  const maxCosine = Math.min(1, (pose.center[1] - 0.12) / 0.88)
  if (Math.cos(pose.angle) > maxCosine) {
    const sign = Math.sin(pose.angle) < 0 ? -1 : 1
    const constrained = sign * Math.acos(maxCosine)
    pose.angle += Math.atan2(Math.sin(constrained - pose.angle), Math.cos(constrained - pose.angle))
  }
  return pose
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
function action(value: unknown): value is SaberAction {
  if (!record(value)) return false
  return (
    (value.color === 0 || value.color === 1) &&
    (value.cell === null ||
      (Number.isInteger(value.cell) && Number(value.cell) >= 0 && Number(value.cell) < 12)) &&
    (value.cut === null ||
      (Number.isInteger(value.cut) && Number(value.cut) >= 0 && Number(value.cut) < 4))
  )
}

/** Independent of React/Three so a future mesh rig can consume the same poses. */
export class FlyMotionController {
  private tracks: [HandTrack, HandTrack]
  private lastStep = -1
  private lastT = -Infinity
  private received = -Infinity
  private pausedAt: number | null = null
  private offset = 0
  private listeners = new Set<() => void>()
  readonly slot: 0 | 1

  constructor(slot: 0 | 1 = 0) {
    this.slot = slot
    this.tracks = [this.restTrack(0), this.restTrack(1)]
  }
  private restTrack(color: SaberColor): HandTrack {
    return { action: { color, cell: null, cut: null }, from: idle(color), changed: 0, strike: 0 }
  }
  private time(now: number) {
    return (this.pausedAt ?? now) - this.offset
  }
  private notify() {
    this.listeners.forEach((listener) => listener())
  }
  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
  reset() {
    this.tracks = [this.restTrack(0), this.restTrack(1)]
    this.lastStep = -1
    this.lastT = this.received = -Infinity
    this.pausedAt = null
    this.offset = 0
    this.notify()
  }
  setPaused(paused: boolean, now = performance.now() / 1000) {
    if (paused && this.pausedAt === null) this.pausedAt = now
    else if (!paused && this.pausedAt !== null) {
      this.offset += now - this.pausedAt
      this.pausedAt = null
    }
    this.notify()
  }
  /** Returns false for unrelated, malformed, duplicate, old, or other-slot messages. */
  acceptMessage(message: unknown, now = performance.now() / 1000): boolean {
    if (!record(message)) return false
    if (message.type === 'ready') {
      this.reset()
      return true
    }
    if (message.type === 'ended' && message.slot === this.slot) {
      this.release(now)
      return true
    }
    if (
      message.type !== 'frame' ||
      message.slot !== this.slot ||
      !record(message.render_state) ||
      this.pausedAt !== null
    )
      return false
    const frame = message.render_state
    if (
      !Number.isFinite(frame.t) ||
      Number(frame.t) < 0 ||
      !Number.isInteger(frame.step) ||
      Number(frame.step) < 0 ||
      !Array.isArray(frame.sabers) ||
      frame.sabers.length < 1 ||
      frame.sabers.length > 2 ||
      !frame.sabers.every(action) ||
      new Set(frame.sabers.map((s) => s.color)).size !== frame.sabers.length ||
      Number(frame.step) <= this.lastStep ||
      Number(frame.t) < this.lastT
    )
      return false
    const t = this.time(now)
    const poses = this.sample(now).hands
    const stale = t - this.received > STREAM_TIMEOUT
    for (const color of [0, 1] as const) {
      const next = frame.sabers.find((s) => s.color === color) ?? { color, cell: null, cut: null }
      let previous = this.tracks[color]
      const end = previous.strike + STROKE_SECONDS
      if (previous.pending && t >= end) {
        previous = {
          action: previous.pending,
          from: poseForTrack({ ...previous, pending: undefined }, end),
          changed: end,
          strike: end,
        }
        this.tracks[color] = previous
      }
      // A one-tick cut must finish even when the next policy frame is no-cut/idle.
      // Keep only the latest non-cut target, never a growing queue of stale actions.
      if (
        !stale &&
        previous.action.cell !== null &&
        previous.action.cut !== null &&
        next.cut === null &&
        t < end
      ) {
        previous.pending = { ...next }
        continue
      }
      const changed =
        stale ||
        previous.pending !== undefined ||
        next.cell !== previous.action.cell ||
        next.cut !== previous.action.cut
      // Sustained cut commands repeat at a readable cadence instead of restarting at 30 Hz.
      const repeat = next.cell !== null && next.cut !== null && t - previous.strike >= 0.28
      if (changed || repeat)
        this.tracks[color] = { action: { ...next }, from: poses[color], changed: t, strike: t }
    }
    this.lastStep = Number(frame.step)
    this.lastT = Number(frame.t)
    this.received = t
    this.notify()
    return true
  }
  /** Call on disconnect/end; ease back from the current pose without continuing stale cuts. */
  release(now = performance.now() / 1000) {
    const t = this.time(now)
    const poses = this.sample(now).hands
    if (this.pausedAt !== null) {
      this.offset += now - this.pausedAt
      this.pausedAt = null
    }
    this.tracks = [0, 1].map((color) => ({
      action: { color: color as SaberColor, cell: null, cut: null },
      from: poses[color],
      changed: t,
      strike: t,
    })) as [HandTrack, HandTrack]
    this.received = t
    this.notify()
  }
  sample(now = performance.now() / 1000): FlyPose {
    const t = this.time(now)
    const staleAge = Math.max(0, t - this.received - STREAM_TIMEOUT)
    const hands = this.tracks.map((track, color) => {
      const rest = idle(color as SaberColor)
      const pose = poseForTrack(track, t)
      return blend(pose, rest, smooth(staleAge / RETURN_SECONDS))
    }) as [HandPose, HandPose]
    return { hands, time: t, active: this.lastStep >= 0 && staleAge === 0 }
  }
}

export const flyMotion = new FlyMotionController()
// Isolated from live sessions so art review never writes into policy state.
export const previewMotion = new FlyMotionController()

/** Attach to an existing session socket. Does not open a connection or start a policy. */
export function attachFlyMotionStream(
  socket: Pick<WebSocket, 'addEventListener' | 'removeEventListener'>,
  controller = flyMotion,
) {
  controller.reset()
  const message = (event: MessageEvent) => {
    if (typeof event.data !== 'string') return
    try {
      controller.acceptMessage(JSON.parse(event.data))
    } catch {
      /* Binary/other consumers own their messages. */
    }
  }
  const close = () => controller.release()
  socket.addEventListener('message', message as EventListener)
  socket.addEventListener('close', close)
  socket.addEventListener('error', close)
  return () => {
    socket.removeEventListener('message', message as EventListener)
    socket.removeEventListener('close', close)
    socket.removeEventListener('error', close)
    controller.release()
  }
}
