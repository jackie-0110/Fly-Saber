# Fly Saber

A simulated fruit fly brain, built from the real MaleCNS connectome, plays **Fly Saber** (a rhythm game) from its own visual input while you watch its neurons fire and change its brain live: lesion cell types, trigger neurons, blind an eye, scramble its wiring.

> Work in progress for SwampHacks. Full spec: [`FLY_SABER_HANDOFF.md`](FLY_SABER_HANDOFF.md). Who builds what: [`FLY_SABER_TEAM_SPLIT.md`](FLY_SABER_TEAM_SPLIT.md).

## Stack

| Layer | Choice |
|---|---|
| Brain model, sim, training | Python 3.11, JAX + optax (BC, then PPO), HiPerGator via SLURM |
| Data | pandas, pyarrow, neuprint-python (MaleCNS), FlyWire Codex |
| Server | FastAPI + WebSockets (uvicorn) |
| Front end | React + Vite + TypeScript, three.js via @react-three/fiber and drei, Zustand, Tailwind |
| Quality | ruff, mypy, pytest, ESLint, Prettier |
| Deploy | Docker on Vultr |

## Team setup

Prereqs: [`uv`](https://docs.astral.sh/uv/) and Node 20+.

```bash
git clone git@github.com:jackie-0110/Fly-Saber.git && cd Fly-Saber
make setup          # uv sync (installs Python 3.11 + deps) and npm install in frontend/
cp .env.example .env  # add your neuPrint token, Gemini key if you need them
make test lint      # should pass on a fresh clone
```

Daily commands: `make serve` (API on :8000), `make frontend` (front end on :5173), `make test`, `make lint`, `make fmt`.

## Layout

```
frontend/    React + three.js app: game scene, 3D brain, controls       (Person 3)
fly_saber/   The game: JAX env, fly eye, scripted expert, song maps     (Person 2)
brain/       Connectome pipeline and the rate network model             (Person 1)
training/    BC, PPO, evaluation, HiPerGator SLURM scripts              (shared, per file)
server/      FastAPI + WebSocket live sessions, narrator, replays       (Person 2, 3)
configs/     YAML configs for every component
scripts/     Data, map, and replay CLI entry points
tests/       pytest, one subfolder per package
docs/        PROTOCOL (hour 0 contracts), DATA_NOTES, ARCHITECTURE, RESULTS
data/        gitignored raw and processed data (except the demo allowlist)
```

Each folder has its own README with what is in it and who owns it. The handoff spec's section 4 layout (`fly_saber/...`) maps onto this as: `connectome/` and `model/` to `brain/`, `sim/` and `maps/` to `fly_saber/`, `train/` to `training/`, `web/` to `frontend/`.

## Who owns what

| Person | Branch prefix | Folders | Docs |
|---|---|---|---|
| 1: Brain and RL | `p1/` | `brain/`, `training/ppo.py`, `training/checkpoints.py`, `training/slurm/` | `DATA_NOTES.md`, groups and activity sections of `PROTOCOL.md` |
| 2: Game, Imitation, Server | `p2/` | `fly_saber/`, `training/bc.py`, `server/` (except narrator, replay) | map, `render_state`, and WebSocket sections of `PROTOCOL.md` |
| 3: Experience and Analysis | `p3/` | `frontend/`, `training/evaluate.py`, `server/narrator.py`, `server/replay.py` | `RESULTS.md` |

Every stub file has an `Owner:` line in its docstring.

## Rules

- `main` always runs the demo. Branch as `p1/subgraph`, `p2/parsers`, `p3/saber-scene`, merge small and often.
- Run `make lint test` before merging.
- Hour 0: fill in the contracts in [`docs/PROTOCOL.md`](docs/PROTOCOL.md) before writing real code, so everyone can build against fake data.
- Never commit data, checkpoints, or `.env`. Only allowlisted royalty-free audio ships.
- No em dashes in user-facing text or docs.
