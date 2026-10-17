# Fly Saber frontend

A responsive Fly Saber observation workspace with an interactive 3D neuron explorer, built with React, TypeScript, Vite, Three.js, React Three Fiber, Drei, Tailwind CSS, and shadcn/ui. The game viewport remains a static illustration while the brain view renders authentic connectome coordinates.

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

`npm run test:unit` validates the real brain export, coordinate transform, original neuron index mapping, and malformed-data handling. It uses Node 22.6 or newer with TypeScript stripping.

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
- The brain renders all 19,462 neurons from the committed MaleCNS v1.0 visual-to-motor subgraph as a single point cloud. This subset contains mostly brain locations and eight distant ascending-neuron somas. It does not include full neuron shapes or a continuous ventral nerve cord reconstruction.
- Drag to orbit, right-drag to pan, and scroll to zoom. Touch supports orbit, pinch zoom, and two-finger pan. Camera buttons offer reset, display-axis views, and zoom. Focus the view to rotate with arrow keys, zoom with +/−, and reset with Home.
- Filter by authentic group and cell type. Click a point or use previous/next buttons to inspect its body ID, backend index, type, side, role, transmitter, original coordinates, and position provenance.
- Expand the neuron view to fullscreen and adjust point size/opacity to inspect dense regions. All 19,462 neurons render, occupying 16,528 distinct coordinate triples because some share partner/estimated positions. Transparent points do not hide deeper neurons with depth writes; coincident points still overlap without invented offsets.
- Colors identify cell types. **Neuron firing data must still be received from the backend.** No activity is simulated. See [BACKEND_ACTIVITY.md](./BACKEND_ACTIVITY.md).
- Game fullscreen and the Neurons/Details/Next steps tabs work. Playback stays disabled until a real session is connected.
- The 3D renderer loads separately from the workspace. Loading, malformed data, unsupported WebGL, and context loss have explanatory states and retry actions; neuron metadata remains available if 3D rendering fails.
- No API, WebSocket, model, audio, or connectome download is needed to view the page. Google Fonts enhance typography when available; local font fallbacks keep the UI readable offline.
- No measured neuron counts, firing rates, or scores are fabricated.

The frontend owner's implementation lives in `frontend/`.

The top-right badge checks the backend's real `GET /api/games` response on load and every five seconds after each check, with a four-second timeout. It reports checking, connected, or disconnected. This measures API reachability; live gameplay and neuron activity still require a future session connection. Vite proxies `/api` and `/ws` to the local backend on port 8000. Production hosting must route those paths to the backend as well.

## Brain assets and coordinates

The browser loads `public/data/brain/neurons.json` and `groups.json`, documented in [docs/PROTOCOL.md](../docs/PROTOCOL.md). Array index i remains the backend neuron index even after filtering. IDs must be safe integers in the current schema; future larger IDs need a string export.

Coordinates are micrometers. The display centers the supplied bounds and applies one uniform scale of `2 / longest_extent`, followed by the rigid axis transform `(x, y, z) -> (x, -z, y)`. Raw positions are retained for inspection. Camera presets refer to display axes, not verified anatomical directions. Position provenance distinguishes soma coordinates, copied partner coordinates, and type-mean estimates (with an overall-mean fallback).

## Next implementation

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the backend contract checklist, live and replay transport architecture, Three.js game scene, rotatable 3D brain, audio synchronization, interventions, and milestone acceptance criteria.

shadcn/ui component source lives in `src/components/ui/`, with CLI configuration in `components.json`. Project-specific composition and styling sit outside those primitives. Add future components from this directory with:

```bash
npx shadcn@latest add select
```

## Illustration sources

`public/previews/fly-saber.svg` is a hand-authored vector illustration. The old `public/previews/neural-preview.svg` is retained artwork, no longer used by the neuron explorer. It can be regenerated by a dependency-free Python script:

```bash
python3 scripts/generate_neural_preview.py
```

That script generates artwork only. Its positions and brightness must never become scientific fixtures, neuron metadata, or activity samples. Future data contracts must follow inspected backend files and [docs/PROTOCOL.md](../docs/PROTOCOL.md).
