# frontend/

**Owner: Person 3 (Experience and Analysis).** Spec section 14.

React + Vite + TypeScript, three.js via @react-three/fiber and drei, Zustand, Tailwind. Run with `make frontend` (port 5173); `/api` and `/ws` proxy to the server on port 8000.

| Path | What |
|---|---|
| `src/game/` | Fly Saber three.js scene: track, blocks, sabers, effects, HUD, audio sync |
| `src/brain/` | 3D neuron cloud, Brain Explorer panel |
| `src/controls/` | Lesion, opto, mood sliders, eye controls |
| `src/eyeview/` | Hexagonal fly's-eye view |
| `src/modes/` | Arcade, Versus, You vs. the Fly, Replay |
| `src/net/` | WebSocket client, replay loader |

Lint: `npm run lint` (oxlint). Format: `npm run format` (Prettier).
