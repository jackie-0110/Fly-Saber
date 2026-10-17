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

## Current behavior

- Desktop windows use a viewport-height workspace with a brain sidebar around 30% of the width and a compact timeline. The brain fills the sidebar width, neuron metadata forms a compact grid below it, and session information has its own tab. Narrow screens stack the panels vertically.
- Song information appears above the viewport: title, artist, mapper, difficulty, initial BPM, decoded audio duration, note count, and a source link for community maps.
- The Songs sidebar defaults to Live songs from `GET /api/maps` and players from `GET /api/brains`. Choose a song and player, then press play to open `/ws/session`. Mapper credit, license/attribution and held-out status are shown.
- Community preview searches the unrestricted community catalog by song, artist, mapper, map ID, or map link. It supports pagination and loading a specific difficulty. Songs download directly to the browser; no fly backend is needed, and the server's demo allowlist is not used by this picker.
- The true 3D game scene shows the track, directional notes, bombs, walls, and an articulated fly mesh with resting motion and independent saber reach/cut animations. Orbit, pan, zoom, overview, player view, reset, fly visibility, and fullscreen controls work. The game camera stays independent of the brain camera and playback time.
- Play, pause, restart, scrub forward/backward, adjust speed, and change volume. Timeline bars show chart note density, not an audio waveform. Block positions derive directly from the audio clock; difficulty changes preserve song position. Scrubbing pauses audio while dragging and resumes on release only if it was playing.
- The brain renders all 19,462 neurons from the committed MaleCNS v1.0 visual-to-motor subgraph as a single point cloud. This subset contains mostly brain locations and eight distant ascending-neuron somas. It does not include full neuron shapes or a continuous ventral nerve cord reconstruction.
- Drag to orbit, right-drag to pan, and scroll to zoom. Touch supports orbit, pinch zoom, and two-finger pan. Camera buttons offer reset, display-axis views, and zoom. Focus the view to rotate with arrow keys, zoom with +/−, and reset with Home.
- Filter by authentic group and cell type. Click a point or use previous/next buttons to inspect its body ID, backend index, type, side, role, transmitter, original coordinates, and position provenance.
- Expand the neuron view to fullscreen and adjust point size/opacity to inspect dense regions. All 19,462 neurons render, occupying 16,528 distinct coordinate triples because some share partner/estimated positions. Transparent points do not hide deeper neurons with depth writes; coincident points still overlap without invented offsets.
- Colors identify cell types. **Neuron firing data must still be received from the backend.** No activity is simulated. See [BACKEND_ACTIVITY.md](./BACKEND_ACTIVITY.md).
- The live session forwards real motor frames to the fly animation adapter and shows server accuracy, combo, score, hits and misses. Note events remove judged notes. Pause/resume preserves score; seek/restart resets simulation state at the selected audio timestamp. Live playback stays at 1× because the protocol has no rate negotiation.
- The fly animation adapter accepts the existing backend motor frame protocol. Open `/?flyMotion=preview` for isolated sample movements; see [FLY_ANIMATIONS.md](./FLY_ANIMATIONS.md) for connection and lifecycle instructions. Fly vision and firing activity still await their backend streams. The frontend does not fabricate policy actions or judge hits.
- The 3D renderers load separately from the workspace. Loading, malformed data, unsupported WebGL, and context loss have explanatory states and retry actions. Song selection, audio, timeline, and neuron metadata remain usable without WebGL.
- No backend, model, or connectome download is needed to view the page or review fly animations. Community search/download needs internet access. Google Fonts are optional, with local fallbacks.
- No measured neuron counts, firing rates, or scores are fabricated.

The frontend owner's implementation lives in `frontend/`.

The top-right badge checks the backend's real `GET /api/games` response on load and every five seconds after each check, with a four-second timeout. It reports checking, connected, or disconnected. This measures API reachability. Live playback has its own connecting, playing, paused, ended and error status; neuron activity remains separate. Vite proxies `/api` and `/ws` to the local backend on port 8000. Production hosting must route those paths to the backend as well. For a separate backend, build with `VITE_BACKEND_URL=https://your-backend.example` (REST, audio and WebSocket all use it); that server must permit the frontend origin. Empty/default uses the page origin.

Live sessions wait up to 15 seconds for readiness before starting audio. Server errors and disconnects stop playback and clear stale motor/score state; pressing play reconnects. Community maps stay local and never start a fly session. Public-audio restriction for community preview (milestone D5) is still pending.

To repeat the real integration check against FastAPI serving this build with staged allowlisted maps/audio:

```bash
FLY_SABER_INTEGRATION_URL=http://127.0.0.1:8000 npx playwright test live-integration --workers=1
# Optional: FLY_SABER_INTEGRATION_BRAIN=real (default is expert)
```

This test opens an actual socket and verifies matching map IDs, score/event frames, clock messages, audio progress and pause on desktop/mobile. Regular `npm test` uses isolated REST/socket fixtures and skips this opt-in test.

## Brain assets and coordinates

The browser loads `public/data/brain/neurons.json` and `groups.json`, documented in [docs/PROTOCOL.md](../docs/PROTOCOL.md). Array index i remains the backend neuron index even after filtering. IDs must be safe integers in the current schema; future larger IDs need a string export.

Coordinates are micrometers. The display centers the supplied bounds and applies one uniform scale of `2 / longest_extent`, followed by the rigid axis transform `(x, y, z) -> (x, -z, y)`. Raw positions are retained for inspection. Camera presets refer to display axes, not verified anatomical directions. Position provenance distinguishes soma coordinates, copied partner coordinates, and type-mean estimates (with an overall-mean fallback).

## Next implementation

See [SONG_PREVIEW.md](./SONG_PREVIEW.md) for supported formats, timing, rendering, and the real-time inference integration boundary. [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) covers the remaining backend contracts, live activity, interventions, and replay work.

shadcn/ui component source lives in `src/components/ui/`, with CLI configuration in `components.json`. Project-specific composition and styling sit outside those primitives. Add future components from this directory with:

```bash
npx shadcn@latest add select
```

## Illustration sources

`public/previews/fly-saber.svg` and `public/previews/neural-preview.svg` are retained artwork. Both viewports now use real 3D rendering. The neural illustration can be regenerated by a dependency-free Python script:

```bash
python3 scripts/generate_neural_preview.py
```

That script generates artwork only. Its positions and brightness must never become scientific fixtures, neuron metadata, or activity samples. Future data contracts must follow inspected backend files and [docs/PROTOCOL.md](../docs/PROTOCOL.md).
