# Fly Saber frontend

A responsive Fly Saber observation workspace with a song browser, an audio-synchronized 3D chart preview, and an interactive 3D neuron explorer. Built with React, TypeScript, Vite, Three.js, React Three Fiber, Drei, Tailwind CSS, and shadcn/ui.

## Run locally

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. To check a production build:

```bash
npm run build
npm run preview
```

`npm run lint` checks the frontend with Oxlint.

`npm run test:unit` validates the real brain export, neuron indices and coordinates, chart normalization, ZIP parsing, timing, audio transport, and fly motor animation. It uses Node 22.6 or newer with TypeScript stripping.

`npm run format` formats the frontend source with Prettier; `npm run format:check` verifies it.

Browser regression tests use Playwright on desktop and mobile Chromium:

```bash
npx playwright install --with-deps chromium
npm run build
npm test
```

Tests run against Vite's production preview on port 4173. `npm run test:ui` opens
the interactive runner. HTML/JUnit reports, screenshots, and failure traces appear
in `playwright-report/` and `test-results/`. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for the full CI checks.

## Deploy (Cloudflare Workers)

`wrangler.jsonc` deploys the `flysaber` Worker: the Vite build is served as static
assets with SPA fallback, and `/api/*` and `/ws/*` (including the session WebSocket)
are proxied by `worker/index.ts` to the FastAPI server, so the browser stays on one
origin and the backend needs no CORS.

Workers Builds settings (Settings → Build): root directory `frontend`, build command
empty (wrangler runs `npm run build` itself), deploy command `npx wrangler deploy`,
non-production branch deploy command `npx wrangler versions upload`.

Point the Worker at the backend once; deploys keep it:

```bash
npx wrangler secret put BACKEND_URL   # e.g. https://flysaber-api.example.com
```

Until it is set, `/api/*` returns 503 and the header shows "Backend disconnected".
Test the Worker locally against a running server with
`npm run cf:dev -- --var BACKEND_URL:http://localhost:8000`, or deploy manually with
`npm run deploy`.

## Current behavior

- Desktop windows use a viewport-height workspace with a brain sidebar around 30% of the width and a compact timeline. The brain fills the sidebar width, neuron metadata forms a compact grid below it, and session information has its own tab. Narrow screens stack the panels vertically.
- Song information appears above the viewport: title, artist, mapper, difficulty, initial BPM, decoded audio duration, note count, and a source link for community maps.
- The Songs sidebar searches the unrestricted community catalog by song, artist, mapper, map ID, or map link. It supports pagination and loading a specific difficulty. Songs download directly to the browser; no fly backend is needed, and the server's demo allowlist is not used by this picker.
- The true 3D game scene shows the track, directional notes, bombs, walls, and an articulated fly mesh with resting motion and independent saber reach/cut animations. Orbit, pan, zoom, overview, player view, reset, fly visibility, and fullscreen controls work. The game camera stays independent of the brain camera and playback time.
- Play, pause, restart, scrub forward/backward, adjust speed, and change volume. Timeline bars show chart note density, not an audio waveform. Block positions derive directly from the audio clock; difficulty changes preserve song position. Scrubbing pauses audio while dragging and resumes on release only if it was playing.
- The brain renders all 19,463 neurons and 340,922 directed connections (6,175,540 synapses) from the exact MaleCNS v1.0 visual-to-motor subgraph bundled with `bundles/real_fast/bc_950.npz`. This subset contains mostly brain locations and distant ascending-neuron somas. It does not include full neuron shapes or a continuous ventral nerve cord reconstruction.
- Drag to orbit, right-drag to pan, and scroll to zoom. Touch supports orbit, pinch zoom, and two-finger pan. Camera buttons offer reset, display-axis views, and zoom. Focus the view to rotate with arrow keys, zoom with +/−, and reset with Home.
- Filter by authentic group and cell type. Click a point or use previous/next buttons to inspect its body ID, backend index, type, side, role, transmitter, original coordinates, and position provenance.
- Expand the neuron view to fullscreen and adjust point size/opacity to inspect dense regions. All neurons render because the view preserves original indices and positions, including shared partner/estimated positions. Transparent points do not hide deeper neurons with depth writes; coincident points still overlap without invented offsets.
- The Connections control has Off, Sample (default) and All. `edges.json` loads after the neurons, only while connections are shown, so the 3D view never waits on it. Sample draws an evenly spaced 20,000 of the filtered edges; software WebGL (CI, weak laptops) stalls on the full 340k. With All, every filtered edge renders at rest using indexed line segments that reuse the neuron positions. During orbit, pan, zoom, or camera-button movement, dense views draw an evenly distributed preview of at most 20,000 actual edges at a pixel ratio capped at 1. The full graph and original pixel ratio return after camera motion and damping settle. The connection count reports the preview while active. Only the draw range changes during movement; geometry is not rebuilt or uploaded each frame. Selected-neuron connections remain complete and use separate incoming/outgoing material groups. Group/type filters keep only edges with both endpoints visible. Selecting a neuron traces its visible incoming (blue) and outgoing (gold) connections. Inspector counts include all its incoming/outgoing connections and anatomical synapses, including connections hidden by filters. Lines connect neuron locations; they are not axon reconstructions, and synapse counts are not learned efficacies or live activity.
- Colors identify cell types. **Neuron firing data must still be received from the backend.** No activity is simulated. See [BACKEND_ACTIVITY.md](../docs/frontend/BACKEND_ACTIVITY.md).
- The live session forwards real motor frames to the fly animation adapter and shows server accuracy, combo, score, hits and misses. Note events remove judged notes. Pause/resume preserves score; seek/restart resets simulation state at the selected audio timestamp. Live playback stays at 1× because the protocol has no rate negotiation.
- The fly animation adapter accepts the existing backend motor frame protocol. Open `/?flyMotion=preview` for isolated sample movements; see [FLY_ANIMATIONS.md](../docs/frontend/FLY_ANIMATIONS.md) for connection and lifecycle instructions. Fly vision and firing activity still await their backend streams. The frontend does not fabricate policy actions or judge hits.
- The 3D renderers load separately from the workspace. Loading, malformed data, unsupported WebGL, and context loss have explanatory states and retry actions. Song selection, audio, timeline, and neuron metadata remain usable without WebGL.
- No backend, model, or connectome download is needed to view the page or review fly animations. Community search/download needs internet access. Google Fonts are optional, with local fallbacks.
- No measured neuron counts, firing rates, or scores are fabricated.

## Source layout

- `src/app/`: application entry point, workspace shell and styles.
- `src/features/brain/`: neuron data, rendering, activity buffer and brain controls.
- `src/features/eye/`: eye lattice, live luminance frames and eye-view UI.
- `src/features/game/`: game scene and fly model/motion.
- `src/features/session/`: backend connection, live protocol, audio transport, stores and session UI.
- `src/features/songs/`: chart parsing, community catalog, imports and song selection UI.
- `src/components/`: shared UI primitives and presentation helpers; `src/lib/`: shared utilities.

Feature components live in each feature's `components/` folder. See the
[repository guide](../docs/REPOSITORY.md) for the full project layout.

The top-right badge checks the backend's real `GET /api/games` response on load and every five seconds after each check, with a four-second timeout. It reports checking, connected, or disconnected. This measures API reachability. Live playback has its own connecting, playing, paused, ended and error status; neuron activity remains separate. Vite proxies `/api` and `/ws` to the local backend on port 8000. Production hosting must route those paths to the backend as well. For a separate backend, build with `VITE_BACKEND_URL=https://your-backend.example` (REST, audio and WebSocket all use it); that server must permit the frontend origin. Empty/default uses the page origin.

Live sessions wait up to 15 seconds for readiness before starting audio. Server errors and disconnects stop playback and clear stale motor/score state; pressing play reconnects. Community maps stay local and never start a fly session. Public-audio restriction for community preview (milestone D5) is still pending.

## Brain assets and coordinates

The browser loads `public/data/brain/neurons.json` and `groups.json`, then `edges.json` once connections are shown, documented in [docs/PROTOCOL.md](../docs/PROTOCOL.md). Array index i remains the backend neuron index even after filtering. `edges.json` includes every `pre`, `post`, `weight` (synapse count), and `sign` from the bundle's `subgraph/edges.parquet`; matching ordered body IDs protect against connecting the wrong neurons. IDs must be safe integers in the current schema; future larger IDs need a string export.

Regenerate the committed assets from the trained bundle at the repository root:

```bash
uv run python scripts/web/export_web.py --bundle bundles/real_fast
```

The exporter verifies the bundle manifest's content hashes and the checkpoint's subgraph metadata before writing assets. The checkpoint contains trained neuron and cell-type-pair parameters; connectivity stays fixed during training. The full anatomical table also includes modulatory/unclear-sign connections; it is not limited to the fast signed synapses used by `load_graph`. `scripts/web/export_brains.py` exports a separate strongest-edge sample for mode comparisons; the neuron view uses the complete table instead.

Coordinates are micrometers. The display centers the supplied bounds and applies one uniform scale of `2 / longest_extent`, followed by the rigid axis transform `(x, y, z) -> (x, -z, y)`. Raw positions are retained for inspection. Camera presets refer to display axes, not verified anatomical directions. Position provenance distinguishes soma coordinates, copied partner coordinates, and type-mean estimates (with an overall-mean fallback).

The camera fits once when a dataset loads. Resizing the canvas preserves the user's orbit, pan, and zoom; Reset explicitly refits to the current dimensions. The connection status reserves two text rows so movement updates cannot resize the canvas. Full-detail restoration stays latched until the next gesture or camera command, preventing late damping updates from repeatedly switching connection visibility.

## Next implementation

See [SONG_PREVIEW.md](../docs/frontend/SONG_PREVIEW.md) for supported formats, timing, rendering, and the real-time inference integration boundary. [IMPLEMENTATION_PLAN.md](../docs/frontend/IMPLEMENTATION_PLAN.md) covers the remaining backend contracts, live activity, interventions, and replay work.

shadcn/ui component source lives in `src/components/ui/`, with CLI configuration in `components.json`. Project-specific composition and styling sit outside those primitives. Add future components from this directory with:

```bash
npx shadcn@latest add select
```
