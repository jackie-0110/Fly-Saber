# fly_saber/

**Owner: Person 2 (Game, Imitation, Server).** Spec sections 7 and 8.

The game itself: the pure JAX environment the fly trains in. No graphics here; `frontend/src/game/` draws the same state in three.js.

| Path | What |
|---|---|
| `env.py` | Fly Saber env: notes on a 4x3 grid, two sabers, scoring, `render_state` |
| `eye.py` | Hexagonal ommatidia lattice, analytic looming rendering, eye controls |
| `experts.py` | Scripted perfect player (must score near 100 on every demo map) |
| `maps/` | BeatSaver client, v2 and v3 parsers, normalizer, adapter, autogen (stretch) |
| `runner.py` | Fly Runner, stretch only (cut for 3 people) |

Configs: `configs/saber.yaml`. Scripts: `scripts/fetch_maps.py`. Demo map licenses: `data/maps/demo_allowlist.json`.
