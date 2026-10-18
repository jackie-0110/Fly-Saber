# Fly movement animations

The procedural fly rig now supports the current backend motor contract: independent red/left and blue/right sabers, reaching all 12 cells, up/down/left/right cuts, holding a cell with no cut, lowering an idle saber, and returning to rest. The front legs follow their saber hilts; the body shifts its weight and the wings respond to movement. Ambient breathing and wing motion are decorative, independent of policy actions, and disabled with `prefers-reduced-motion`.

Open `/?flyMotion=preview` to review movements without a backend. Choose a movement, saber, and target cell, or cycle through all movements. Pause freezes the pose. This preview has its own controller and never changes songs, policy state, or scores. Normal pages show only resting motion until real frames arrive.

## Connect the trained model

The adapter matches [`docs/PROTOCOL.md`](../PROTOCOL.md) and `src/server/protocol.py`. `frontend/src/features/session/session-client.ts` now owns the live socket and forwards validated frames to `flyMotion.acceptMessage`, mirroring pause, seek, reset, end and disconnect. No additional attachment is needed in the app. A standalone consumer can attach it to its own socket:

```ts
import { attachFlyMotionStream, flyMotion } from '@/features/game/fly-motion'

const detachMotion = attachFlyMotionStream(sessionSocket)
// Existing session code sends start/play/clock and owns audio synchronization.
// Mirror its pause state locally when sending { type: 'pause', paused }:
flyMotion.setPaused(paused)
// Before a seek/new play timeline, clear step ordering:
flyMotion.reset()
// On teardown:
detachMotion()
```

Alternatively, forward parsed messages to `flyMotion.acceptMessage(message)`. It accepts `ready`, `frame`, and `ended`; unrelated messages and binary activity are ignored. The default controller renders slot 0. For another slot, construct `new FlyMotionController(1)` and pass that controller to both the adapter and `<FlyModel controller={controller} />`.

`frame.render_state` supplies `t`, `step`, and `sabers: [{ color, cell, cut }]`. Cells are `layer * 4 + lane` (0–11), or `null` for idle. Cut is 0 up, 1 down, 2 left, 3 right, or `null`. Send wire-format nulls, not the internal JAX sentinels 12/4. A missing second saber rests naturally for single-saber policies. Array order does not determine color.

Frames are validated before changing pose. Duplicate/out-of-order steps, backwards timestamps, invalid cells/directions, and other slots are ignored. `ready` resets ordering; call `reset()` before a seek or new `play` because the server restarts its step counter. `setPaused()` freezes animation time and stops accepting new frames until resumed. Close/error/detach eases the rig to rest, including when disconnected while paused. Without new frames for 350 ms, the rig eases to rest over 250 ms.

Reaches blend from the previous pose. A single-tick cut finishes before following the latest no-cut/idle target; new cuts can interrupt immediately. A 140 ms stroke passes the blade midpoint through its selected grid center. Persistent identical cuts repeat at most every 280 ms instead of restarting at every 30 Hz frame; changing cells/cuts or returning to no-cut re-arms immediately. These are display transitions smoothed from frame arrival time, not hit detection or an assertion of exact audio timing. The backend remains authoritative for judging actions and outcomes.

The controller and mesh are separate: replace `fly-model.tsx` or bind `sample().hands` to a later asset's bones without changing the protocol adapter. The current stylized forelegs stretch to cover the grid. The current backend has no walking, flight, duck, dodge, or diagonal action heads, so those are not synthesized from notes or wall positions. The app opens the WebSocket when playing a server-catalog song with an available player. Community preview and animation review open no session.

Validation: `npm run build`, `npm run lint`, `npm run test:unit`, and Playwright checks in `e2e/fly-motion.spec.ts`.
