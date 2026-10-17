# FlySaber frontend implementation plan

## Scope and current state

This plan follows [`FLYSABER_HANDOFF.md`](../FLYSABER_HANDOFF.md), especially sections 7, 13, and 14, and the supplied reference image. The handoff describes a fly playing from visual input, with interventions that affect its behavior. The frontend must communicate that goal without claiming that this static preview demonstrates it.

The implementation lives in the repository's canonical `frontend/` directory. The handoff's older `web/` path was superseded by the team reorganization; `make frontend` runs this client. Preserve the existing Vite `/api` and `/ws` development proxies for integration with `server/`.

Implemented now:

- React, TypeScript, and Vite with shadcn/ui components and a dark grey workstation design system.
- A workstation layout with a compact toolbar, large Fly Saber viewport, right-hand brain/session inspector, and bottom timeline/readouts, stacking vertically on small screens.
- Two original, local SVG images. Neither is a recorded session, scientific reconstruction, or source of test data.
- Honest static-preview labels, inactive playback, unreported scores, fullscreen viewing, and Activity/Details/Next steps inspector tabs.
- No network session, fake firing animation, audio, game simulation, or 3D camera interaction.

The point cloud is a decorative illustration of a brain and ventral nerve cord. Do not use its positions, point count, or colors as scientific data. Display only the anatomical structures supplied in the eventual backend asset.

## Visual direction

Treat the app as an arcade observation workspace: the game is the main focus; the brain provides a connected scientific view. Follow the spatial-analysis reference: a compact toolbar, dominant viewport, right inspector, and bottom timeline/readouts. Use dark grey and charcoal surfaces, restrained borders, compact typography, and muted teal/coral accents. Avoid commercial game artwork, names, music, or branding. The original Fly Saber illustration uses broad brightness patterns, rather than small arrows alone, in anticipation of the fly-visible direction encoding.

Use the checked-in shadcn/ui Button, Badge, Card, Tooltip, Tabs, and Separator components for consistent interaction and accessibility. Add Select, Switch, Slider, and Sheet when their associated features actually work. Keep backend-dependent actions unavailable until their server capability is present. Clearly distinguish preview, connecting, live, paused, disconnected, and recorded replay states.

## 1. Agree on real contracts before adding a connection

[`docs/PROTOCOL.md`](../docs/PROTOCOL.md) now contains draft map, render-state, REST, and WebSocket contracts, with Pydantic models in `server/protocol.py`. The server runs a fake simulation; authentic neural activity, interventions, and brain inference are still pending. Build against these inspected contracts, confirm remaining details with their owners, and never treat the preview SVG as a fixture.

Current integration facts from the checked-in contracts:

- Fetch the full normalized map over REST. Note distance is `(time_sec - song_t) * note_speed_mps`; grid cells use `layer * 4 + lane`. Frames contain saber cells/cuts and score rather than every note position.
- Start with `start`, wait for `ready`, then send `play {song_t}` when audio starts. Send `clock {song_t}` about once a second; `pause` carries a boolean. Preserve `slot` for each brain.
- Binary messages have an 8-byte prefix: kind, slot, two reserved bytes, then a little-endian uint32 simulation step. The eye payload is described; neuron rates and cell-type mean payloads remain TODO.
- Subgraph indices are neuron row numbers. `neurons.parquet` includes `body_id`, `type`, `side`, and `x/y/z` in 8 nm voxels, with `position_inherited` metadata and some NaN positions. Preserve the index mapping when excluding missing points; disclose inherited positions. Do not fabricate positions for missing neurons.
- `set_interventions`, `human_action`, and `narrate` currently return errors. Their controls must remain unavailable.

Inspect actual backend responses and exported assets with the server and connectome owners. Confirm or complete these agreements in `docs/PROTOCOL.md`:

| Contract | Questions to settle |
| --- | --- |
| Render state | Stable object IDs, spawn/despawn rules, positions versus map-derived positions, units, axes, origin, lane/layer mapping, hit plane, saber poses, cut directions, hit events, score and simulation timestamp. |
| Brain asset | Exact source/version, included structures, ordered neuron IDs, point positions, units, axis orientation, cell types, group membership, coordinate-to-simulation transform, asset size, and cache/version identifiers. |
| Activity | Exact neuron index order, binary framing, stream/session identity, sequence numbers, timestamps, quantization, scale/offset, missing values, rate units, and full versus partial updates. |
| Session lifecycle | Start acknowledgement, time zero, model/checkpoint/brain/map identities, pause/resume/reset, errors, reconnect, unavailable capabilities, and intervention acknowledgement. |
| Audio/maps | Demo allowlist, file URLs, duration, timing offset, difficulty, mapper credit, license, and whether a song is held out from training. |
| Replay | Versioned recording container, asset references, frames, activity, events, timebase, and which interventions were recorded. |

Use stable string IDs when upstream IDs may exceed JavaScript's safe integer range. Require an explicit neuron index mapping and matching asset version before applying an activity buffer. Reject incompatible lengths or stale sessions; never silently light up the wrong neurons. Do not label quantized model rates as Hz unless the backend establishes that interpretation and conversion.

Done when: genuine small sample files are available, their fields are inspected and documented, and the team agrees on axes, neuron ordering, and timebase.

## 2. Separate transport, application state, and rendering

Keep the existing preview as the initial and failure state. Evolve toward:

```text
src/
  components/ui/           checked-in shadcn/ui primitives
  components/             workspace shell, statuses, transport
  game/                   GameViewport, Track, Notes, Sabers, effects
  brain/                  BrainViewport, NeuronCloud, CameraControls, Explorer
  controls/               interventions, groups, session controls
  eyeview/                both-eye mosaic when data exists
  net/                    verified protocol types, decoders, socket lifecycle
  replay/                 recording loader and playback transport
  state/                  session and UI stores, frame/activity buffers
  lib/                    shared helpers and coordinate/clock adapters
public/previews/           static fallbacks
```

Use Zustand for low-frequency session/UI state. Keep frame and activity buffers outside React component state. Define one transport-independent data interface after examining real data, then adapt both live streams and recorded sessions into it. Local camera state must remain independent of the simulation.

Validate external messages at the boundary. Bound interpolation buffers, use capped reconnect backoff, dispose listeners and sockets on unmount, and cancel callbacks on session changes. A disconnection should freeze the last valid view, show its age, and offer an explicitly labeled replay when available. A static image is not a replay.

Done when: recorded and live adapters drive the same UI and a dropped connection cannot silently continue as live.

## 3. Replace the Fly Saber image with a real 3D scene

Use `three`, `@react-three/fiber`, and `@react-three/drei`, in versions compatible with the installed React major. These are planned dependencies, not required by today's static images. Lazy-load the renderer so the dashboard and fallback still load without WebGL.

1. Add an R3F Canvas inside the existing game viewport, retaining its accessible HTML heading, HUD, fullscreen button, and transport.
2. Build an original track, floor, perspective camera, four lanes, three layers, and hit plane against the agreed coordinate transform. Render blocks with instancing and use pooled short-lived hit effects. Use a simple original fly mesh first; a small, licensed GLB can replace it later.
3. Match the backend's actual fly-visible brightness/shape encoding. Match the same handedness, cell positions, cut direction, and saber color. Human-readable arrows may supplement broad patterns, but must not suggest the fly receives information absent from its visual input.
4. Drive saber poses, hits, misses, bombs, score, combo, and accuracy from authoritative simulation state/events. Interpolate visual poses between snapshots; do not simulate policy decisions or award hits in the browser.
5. Add allowlisted song selection with mapper and license credits. Require a user gesture to start audio. Preserve an image fallback for loading, asset failure, WebGL failure, and static preview.

### Movable game camera

The game view must be a true 3D scene with a movable camera, in addition to the independently rotatable brain. Add Drei OrbitControls scoped to the game canvas: drag to orbit, modified/right drag to pan, and wheel/pinch to zoom. Provide keyboard equivalents and shadcn buttons for default third-person, track overview, and reset view. Keep the fly and incoming notes within useful framing with sensible distance/target limits. Offer a follow-fly view later only if backend pose data supports it.

Changing the spectator camera must not change the fly's visual input, ommatidia camera, policy actions, game coordinates, or the brain camera. Keep both camera states outside streamed simulation state and preserve them through pause and activity updates. If human play is added later, give spectator camera navigation a distinct interaction mode so it does not consume gameplay inputs. Preserve Escape for fullscreen exit and keep controls scoped so page scrolling works outside each viewport.

### Simulation and audio timing

Follow the current protocol's client audio clock: after `ready`, send `play {song_t}` when audio starts and `clock {song_t}` about once a second. The server catches up or holds its simulation against that timeline. Simulation outcomes remain authoritative; note presentation follows the normalized map and audio song time. Use `AudioContext.currentTime` for smooth audio-aligned presentation between frames. Measure network latency and jitter, buffer briefly, and confirm drift handling with the server owner. Where supported, `getOutputTimestamp()` can relate audio output to the performance clock.

Never advance notes using frame counts. Agree with the server owner on acceptable presentation delay and drift thresholds; handle pause, suspended audio contexts, tab visibility, restart, reconnect, playback rate, and seeking explicitly. A hard jump may require audio restart rather than replaying old hit effects. Test a full actual song, not just an opening segment.

Done when: actual streamed/recorded notes and saber poses match the simulation, audio remains aligned over a full demo song, the game camera can orbit/pan/zoom/reset independently of the brain, and the fallback remains usable.

## 4. Replace the neural image with a rotatable 3D brain

### Data and rendering

Load authentic coordinates and metadata once per versioned brain asset. Convert upstream coordinates into one centered group with a documented axis transform and uniform scale. Preserve the raw coordinates and IDs for inspection. Fit the camera from the asset's bounding box; never reshape the anatomy to fit the panel.

Render the planned 5,000 to 20,000 neuron subgraph as a single `THREE.Points`/BufferGeometry cloud, with attributes for position, type color, and activity. Avoid a mesh or React component per neuron. Update activity attributes at the stream rate, mark buffers dirty, and interpolate brightness in the render loop. Keep a subdued type-colored baseline and an accessible activity legend. Add a shader only if profiling justifies it. Render connectivity on selection with an edge budget, rather than displaying millions of edges.

The current export uses soma positions in 8 nm voxels, with some positions inherited from a partner and some missing. Confirm the browser export retains those provenance flags and the original row-index mapping. Points are a spatial summary, not full neuron morphology. If detailed morphology is needed later, fetch selected neuron skeletons or meshes on demand, with provenance and memory limits.

### User rotation and navigation

- Add Drei `OrbitControls` to the brain viewport with damping, left-drag orbit, wheel/pinch zoom, and documented touch gestures. Limit zoom using the actual bounding sphere.
- Allow full orbit so the depth is visible. Do not rotate a flat image with CSS or animate the camera automatically.
- Keep pointer input confined to the brain viewport so game controls and page scrolling work outside it. Provide a visible interaction hint only once rotation works.
- Add shadcn buttons for reset camera, front, side, and top presets, plus keyboard-accessible rotation and zoom. Name anatomical presets only after confirming coordinate orientation with the backend team.
- Preserve the current camera when new activity arrives or the user pauses the session. Camera movement must never change model state.
- Initially show all supplied neurons. Later add cell-type/group filters, hover labels, and click selection with an appropriate raycast threshold or spatial index. Distinguish clicks from orbit drags. Link selected indices back to stable neuron IDs and explorer metadata.
- Display the actual loaded subgraph count and dataset version. Never imply the full source connectome is loaded when showing a subset.

Done when: genuine positions retain depth from every angle, mouse/touch/keyboard users can explore and reset the view, and a known activity sample highlights the correct IDs without resetting the camera.

## 5. Wire interventions and replay before expanding scope

Must ship, following the handoff:

1. Live Fly Saber plus authentic neural activity and orbit controls.
2. Lesion toggles populated from real `groups.json`, with pending/applied/error states tied to server acknowledgement and a reset action.
3. Real versus scrambled comparison using the same map and seed, with independently identified streams and honest scores.
4. Offline playback of a real recorded session through the same rendering interface. Clearly label recording identity and active recorded interventions. Arbitrary interventions must be disabled in recorded replay.

Target features after those work: Brain Explorer, optogenetics, mood sliders, eye controls and the two-eye mosaic, then narration. Describe biological functions using approved metadata. Do not invent explanations or claim changes to game behavior before the server acknowledges and produces them. Defer additional games and hardware integration.

## Performance, accessibility, and verification

Target at least 30 fps on an agreed laptop while the handoff's intended 30 Hz game frames and 15 Hz per-neuron activity arrive. Measure actual rates instead of presenting those targets as runtime facts. Start with two independent canvases for isolation; profile CPU/GPU and context cost before considering a shared canvas with scoped views.

Cap device pixel ratio, avoid expensive shadows and bloom initially, reuse geometry/materials, and release GPU resources on brain/session changes. Bound point/edge/effect budgets and network buffers. Reduce work when the tab is hidden. Render an idle paused brain on demand while preserving orbit interaction. Keep a static fallback with meaningful alt text and a retry action for context loss or unsupported WebGL.

Use shadcn accessibility primitives, visible keyboard focus, tooltips that also appear on focus, adequate text contrast, and reduced-motion preferences. Do not make color the only activity/status cue. Preserve mobile page scrolling outside the rotatable canvas.

Verify at each milestone:

- TypeScript build and relevant lint checks, with a clean production build.
- Desktop and narrow layouts, keyboard interaction, fullscreen entry/exit, fallback/error states, and no horizontal overflow.
- Coordinate adapter and decoder tests against actual backend fixtures, including large IDs, mismatched neuron ordering, truncated binary payloads, stale sessions, and reordered frames.
- Integration tests for start/pause/reset/reconnect and intervention acknowledgement; audio drift and replay seek/pause/resume checks against a recorded session.
- Manual game-camera orbit/pan/zoom/reset, independent brain-camera orbit/presets, point selection, resize, tab suspension, context loss, and sustained frame-rate/memory checks with the real neuron budget. Verify spectator camera changes never affect the fly's-eye input or policy.
- Offline replay with the API stopped. No external font or asset availability should be required for basic playback or readable UI.

## Official implementation references

- [shadcn/ui with Vite](https://ui.shadcn.com/docs/installation/vite): checked-in components and Tailwind integration.
- [React Three Fiber introduction](https://r3f.docs.pmnd.rs/getting-started/introduction): React renderer and major-version compatibility.
- [Drei controls](https://drei.docs.pmnd.rs/controls/introduction): camera control lifecycle and demand rendering support.
- [Three.js documentation](https://threejs.org/docs/): Points, BufferGeometry, instancing, raycasting, and resource disposal.
- [AudioContext.getOutputTimestamp](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/getOutputTimestamp): mapping audio output timestamps to the performance clock.
