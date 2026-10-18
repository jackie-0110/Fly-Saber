<div align="center">
<img src="https://github.com/user-attachments/assets/beb24fe1-3ba2-4238-ab3a-9a8a21ecba2a" alt="Fly Saber banner" width="100%">

*A fruit fly brain built from the real connectome plays a rhythm game through its own eyes, and you can rewire it while it plays*

[![CI](https://github.com/jackie-0110/Fly-Saber/actions/workflows/ci.yml/badge.svg)](https://github.com/jackie-0110/Fly-Saber/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/demo-flysaber.jackie--courtney.com-brightgreen.svg)](https://flysaber.jackie-courtney.com)
![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-orange)
![Built For](https://img.shields.io/badge/built%20for-SwampHacks-blue)

</div>

<div align="center">
  <img width="3008" height="1632" alt="image" src="https://github.com/user-attachments/assets/0f560e5c-306a-40c5-a64a-de376c855294" alt="The trained fly playing Sneaky Adventure live, with its neuron cloud lit by activity" width="100%">
  <br>
  <sub><b>Live play.</b> The trained fly slashing notes in Sneaky Adventure while its 19,463 neurons light up by how much their firing changes from rest.</sub>
</div>

## Overview

Fly Saber simulates **19,463 neurons and 340,922 connections** from the visual-to-motor pathway of the [MaleCNS](https://male-cns.janelia.org/) connectome and runs them on a rhythm game. Blocks approach in a 4x3 grid to the beat, and the fly must slash the right ones in the right direction. Pick almost any BeatSaver map, including songs the fly has never seen.

Earlier fly-brain game demos are watch-only, and some replay recorded movements instead of reacting to what is on screen. Here the fly **plays from vision alone**: it sees only a hexagonal compound-eye rendering of the scene (437 columns), gets no game state, and swings two sabers with its own descending neurons at 30 Hz. While it plays you can lesion cell types, blindfold either eye, move neuromodulator mood sliders, or swap in a scrambled brain, then watch the score and the neuron cloud respond.

## How It Works

Fly Saber runs the following pipeline, from raw connectome to a live browser session:

1. **Extract** -- `src/brain/connectome/` pulls a visual-to-motor subgraph from MaleCNS v1.0: photoreceptors through lamina, medulla and lobula to the visual projection neurons, then the descending neurons that drive movement. That is 19,463 neurons of 1,633 cell types and 6.2 million synapses, with edge signs from neurotransmitter identity
2. **Model** -- `src/brain/model/brain.py` is a leaky rate network on that wiring. The connections stay fixed. Training sets each neuron's time constant, bias and output scale, plus a linear readout from the descending neurons to the sabers
3. **Modulate** -- octopamine, dopamine and serotonin neurons scale the gain of the neurons they reach, which is how the mood sliders work
4. **See and act** -- `src/fly_saber/` is a JAX game environment. The fly's eye samples the scene in 437 hexagonal columns (217 left, 220 right), luminance plus color, wired to the brain's own photoreceptor columns. Each saber picks a cell and a cut direction
5. **Train** -- `src/training/bc.py` uses behavior cloning: the fly imitates a scripted expert across community BeatSaver charts, trained on HiPerGator GPUs
6. **Serve** -- `src/server/` steps the trained brain at 30 Hz per session over a WebSocket and streams saber actions, judged notes and per-neuron activity. Lesions, blindfolds and moods apply mid-song without recompiling
7. **Render** -- `frontend/` (React and three.js) plays the song, draws the game, and lights the neuron cloud

The wire format is in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

### Four brains to choose from

| Brain | What it is |
|---|---|
| Real wiring | The actual connectome. This is the demo fly |
| Shuffled signs | The real connections, with excitatory and inhibitory labels scrambled across neurons |
| Shuffled within layers | Each neuron keeps its layer but gets random partners inside it |
| Fully rewired | A degree-preserving shuffle of the entire graph |

## Results

Greedy play, no expert help. Hit rate is notes hit over notes judged, on songs held out from training. Full tables, learning curves and caveats are in [`results/README.md`](results/README.md).

**It plays from vision.** Live controls on the demo fly (`bundles/real`, 64 windows of 10 s):

| Change | Held-out hit rate |
|---|---|
| None | 32.9% |
| Looming detectors (LPLC2, LC4, LPLC1, LPLC4) lesioned | 14.1% |
| One eye blindfolded | 7.1% to 7.6% |
| Motion detectors (T4, T5) lesioned | 2.1% |
| Both eyes blindfolded | 0% |

**The wiring matters, with a twist.** Seed 0, 256 windows of 10 s:

| Brain | Held-out hit rate |
|---|---|
| Real wiring, brain trained | 31.4% |
| Real wiring, readout only | 33.9% |
| Partners shuffled within layers | 51.8% |
| Excitation and inhibition shuffled | 20.0% |

Scrambling excitation and inhibition cuts performance by a third, so the sign structure carries real information. Shuffling partners within layers *beats* the real wiring. The likely reason is that the real circuit compresses vision into signals that matter to a fly (looming anywhere in view) and discards the where-on-screen detail this game rewards. The scripted expert scores 100% on every demo map.

## Architecture

```text
 BeatSaver map ──► fly_saber env ──► compound eye (437 columns)
                                          │
                                          ▼
   React + three.js  ◄── WebSocket ──  FastAPI session ──► brain (19,463-neuron rate network)
   (game, neuron cloud)   30 Hz          lesions, moods         │
                                                                ▼
                                                  descending neurons ──► sabers
```

<sub><b>System architecture.</b> JAX game and brain in the Python backend, streamed over a WebSocket to a React and three.js front end, shipped as one Docker image that CI builds and smoke-tests on every pull request.</sub>

## Getting Started

Try the live deployment at **[flysaber.jackie-courtney.com](https://flysaber.jackie-courtney.com)**, or run it locally:

### Prerequisites

- Node 22 (`nvm use`)
- Python 3.11
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker (optional, for the container build)

No API keys, GPU, or connectome download are needed to play: the trained brains ship in `bundles/`, each with the exact subgraph it was trained on.

### 1. Install and fetch songs

```bash
git clone https://github.com/jackie-0110/Fly-Saber.git && cd Fly-Saber
make setup    # locked Python 3.11 and frontend dependencies
make maps     # download the four CC BY demo songs from BeatSaver
```

### 2. Build and run

```bash
make build    # build the front end into frontend/dist
make serve    # API, WebSocket and front end on http://localhost:8000
```

Open http://localhost:8000, choose a song, press play, and open the **Neurons** tab.

### 3. Or use Docker

```bash
make docker-build
docker run --rm -p 8000:8000 fly-saber:local
```

Deploying to a server is covered in [`docs/CI_CD.md`](docs/CI_CD.md).

## Development

```bash
cd frontend && npx playwright install --with-deps chromium && cd ..   # once
make check    # lint, strict types, pytest, frontend build, desktop and mobile browser tests
```

Daily commands: `make serve` (API on :8000), `make frontend` (Vite dev server on :5173), `make test`, `make lint`, `make fmt`. Tests use small synthetic graphs and maps, so they need no connectome, GPU, credentials or audio.

To rebuild the brain from scratch: `make data` downloads the MaleCNS files (about 1.2 GB, no account needed), `make subgraph` extracts the subgraph, and `make train-bc` trains, or use `infra/slurm/` on HiPerGator.

Pull requests run Python 3.11 and 3.12, the frontend, and a Docker smoke test in which the trained fly plays a full session inside the container. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## Repository Layout

```text
src/                 Python packages: brain, fly_saber, training, server
frontend/            React and three.js app (app, features, components, lib, tests, e2e)
scripts/             Maintenance commands grouped by data, connectome, training, web
infra/               Docker build and smoke fixtures, deployment scripts, SLURM jobs
configs/             YAML config for every component
bundles/             Trained brains with their exact subgraphs
data/                Raw downloads, processed data and song maps
results/             Training runs and control evaluations
docs/                Technical docs, project plans and frontend notes
tests/               Python tests, one folder per package
```

See the [Repository guide](docs/REPOSITORY.md) for module boundaries, commands and file placement.

## Team

| Person | Focus | Owns |
|---|---|---|
| 1 | Brain and RL | `src/brain/`, training runs on HiPerGator, `results/`, hosting |
| 2 | Game, imitation, server | `src/fly_saber/`, `src/training/bc.py`, `src/server/` |
| 3 | Experience | `frontend/`, the demo |

Who builds what in detail: [`docs/project/FLY_SABER_TEAM_SPLIT.md`](docs/project/FLY_SABER_TEAM_SPLIT.md). Branches use the owner's prefix (`p1/`, `p2/`, `p3/`), and `main` always runs the demo.

## Documentation

| Doc | What |
|---|---|
| [`docs/project/FLY_SABER_HANDOFF.md`](docs/project/FLY_SABER_HANDOFF.md) | The full spec |
| [`docs/project/MILESTONE.md`](docs/project/MILESTONE.md) | What stands between us and the MVP |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | Map schema, `render_state`, WebSocket messages, interventions, activity stream |
| [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | What the connectome files actually contain |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component boundaries and conventions |
| [`docs/CI_CD.md`](docs/CI_CD.md) | CI, the container image and deployment |
| [`results/README.md`](results/README.md) | Training runs, baselines and what each control does |

---


## Acknowledgements

- **Connectome:** [MaleCNS v1.0](https://male-cns.janelia.org/), Janelia FlyEM and collaborators, via [neuPrint](https://neuprint.janelia.org/).
- **Demo music:** "Sneaky Adventure", "Le Grand Chase", "Pop Goes the Weasel" and "Fluffing a Duck" by Kevin MacLeod ([incompetech.com](https://incompetech.com)), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Charts by their BeatSaver mappers, credited in the app.
- **Compute:** [HiPerGator](https://www.rc.ufl.edu/about/hipergator/), University of Florida Research Computing.
- Built with [JAX](https://github.com/jax-ml/jax), [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/) and [three.js](https://threejs.org/).

## License

No license has been chosen yet. Until one is added, ask the team before reusing the code. The connectome data and the music carry their own licenses (above).
