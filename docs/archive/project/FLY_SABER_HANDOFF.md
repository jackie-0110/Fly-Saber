# Fly Saber: Coding Agent Handoff

You are building **Fly Saber**, a web app where a simulated fruit fly brain, built from the real connectome, plays 3D browser games while users watch its neurons fire and change anything about its brain in real time.

This document is the full spec. Read all of it before writing code. Work milestone by milestone, keep the must-ship tier working at all times, and verify each milestone before moving on.

---

## 0. Ground rules

1. **Hackathon timing.** This is for SwampHacks (24 hours). Most hackathons ban code written before hacking starts. Do not start writing project code until the event begins. Before the event, the only allowed prep is downloading datasets, reading documentation, and inspecting data formats. Confirm the event rules with the team.
2. **Never invent data schemas.** Connectome exports and Beat Saber map formats vary by version. Always inspect real files first (print columns, keys, row counts) and write code against what is actually there. If a field this doc mentions does not exist, find the equivalent and note the difference in `docs/reference/DATA_NOTES.md`.
3. **No third-party branding.** Do not use the names, logos, art, or audio of Beat Saber, Subway Surfers, or any other commercial game. Our games are **Fly Runner** and **Fly Saber**.
4. **Code quality is judged.** The Best Use of AI/ML prize scores coding conventions. Use type hints, small modules, config files instead of magic numbers, tests for core logic, `ruff` and `mypy` clean, and a strong README.
5. **Always have a demo.** Build the replay fallback early (Milestone 4). A crashed server or training job must never kill the demo.
6. **Do not use em dashes** in any user-facing text, README, or docs written for this project (team preference).

---

## 1. Product summary

### Pitch
"You've seen the viral fly brain play Beat Saber. It was replaying movements. Ours actually sees the blocks, and you get the controls: blind one eye, lesion its motion detectors, flood it with arousal, scramble its wiring, and watch what happens."

### Context (why this is novel)
After the MaleCNS v1.0 connectome (about 166,000 neurons, adult male fruit fly, Google Research and HHMI Janelia) was released, viral demos connected it to Doom, Beat Saber, Minecraft, and Mario 64. These demos are watch-only, and the Beat Saber one did not independently play: its motor output replayed movements. Fly Saber fills both gaps: a fly network that genuinely plays from visual input, plus full interactive control over its brain while it plays.

### Games
| Game | Tier | Description |
|---|---|---|
| **Fly Saber** | **Must ship (the main game)** | Rhythm game in the style of Beat Saber. Blocks approach in a 4x3 grid to the beat; the fly moves its saber and slashes in the right direction. Plays imported community song maps, including maps it has never seen. |
| Fly Runner | Stretch | Endless 3-lane runner sharing the same engine. Secondary game for the cross-game lesion demo. |
| Fly Flight | Moonshot | Fly steers through a tunnel of obstacles. |
| Fruit Fly Ninja | Moonshot | Fruit flies toward the fly; it slashes. Reuses Fly Saber code. |

All games share one engine: objects approach the fly in 3D, and the fly reacts in time. **Fly Saber is the product.** Everything else is optional. Incoming blocks are looming stimuli, which the fly visual system is built to detect, so the game is a natural fit as long as cut direction is encoded in a way fly vision can resolve (Section 7.3).

### Features (all act on the live, playing brain)
1. **Arcade:** pick a game and a brain variant, watch it play with a live 3D neuron view.
2. **Brain Explorer:** click any neuron or cell type to see its identity, connections, known biological function, and live activity trace. Hovering a game object highlights neurons responding to it.
3. **Lesion Mode:** switch off cell types (motion detectors, looming detectors, motor pathways).
4. **Optogenetics Mode:** force a neuron group on (for example trigger the escape pathway mid-song).
5. **Mood sliders:** global modulators such as arousal and reward sensitivity.
6. **Eye controls:** blind left or right eye, blur, reduce facet count, add visual noise. A **fly's-eye view** panel shows the hexagonal mosaic the fly sees.
7. **Versus Mode:** two brains side by side on the same level (real vs. scrambled, MaleCNS vs. FlyWire, fly vs. MLP).
8. **You vs. the Fly:** a human plays the same level with the keyboard beside the fly.
9. **Brain narrator (Gemini):** plain-English explanation of why the fly just failed; natural-language commands mapped to controls.
10. **Leaderboard (stretch):** scores per brain variant per game.
11. **Brain Lab (moonshot):** user-defined rewiring.

### Scope tiers
- **Must ship:** Fly Saber played by the trained fly from visual input, map parser plus normalizer (v2 and v3) with hand-picked royalty-free demo maps, song audio playing in sync, live 3D neuron view, Lesion Mode, real vs. scrambled Versus, replay fallback, deployed public URL.
- **Target:** automated BeatSaver training set (hundreds of maps) with curriculum training and evaluation on held-out maps, Brain Explorer, Optogenetics, mood sliders, fly's-eye view, eye controls, Gemini narrator.
- **Stretch:** "upload a song" auto-mapping, 8 cut directions, You vs. the Fly, FlyWire vs. MaleCNS, leaderboard, Fly Runner.
- **Moonshot:** Fly Flight, Fruit Fly Ninja, Brain Lab, WebXR (play against the fly in a headset), physical control box.

Decision point at **hour 14**: decide which tier the team is in and cut the rest.

---

## 2. Prize strategy (drives some requirements)

| Prize | Priority | What the build must include |
|---|---|---|
| Best Overall | Core | Polished live demo and clear story. |
| Best Use of AI/ML | Core | Custom model, training (BC plus PPO), evals. **Public repo linked on Devpost with a README.** Clean code conventions. Results (learning curves, real vs. scrambled, unseen-map eval) in the README. |
| Best Entertainment & Creative Tech | Core | Games, music, interactivity: Fly Saber with song maps. |
| MLH Best Use of Vultr | Freebie | Host inference server and web app on Vultr (GPU instance if needed). Document it in the README. |
| MLH Best Domain (GoDaddy Registry) | Freebie | Register a domain for the live URL. |
| MLH DEV Writing Challenge | Freebie | Post-event writeup (team task, not agent). |
| Best UI/UX | Stretch | Polish of 3D brain, control panel, fly's-eye view. |
| MLH Best Use of Gemini API | Stretch | Brain narrator (Section 9). |
| MLH Best Use of Tiger Data | Optional | Only if replays or leaderboard are built: store sessions, scores, activity time series (Section 10). |
| Best Hardware Hack | Optional | Physical control box (Section 11). Only with spare people after hour 14. |

Skip: Roblox, Nessie, Solana, PlanIT, DeepSpace, Auth0, Least Vibe Coded. Team must check how many tracks a project may enter.

---

## 3. Tech stack

- **Python 3.11+**, managed with `uv` (or `pip` plus `requirements.txt` if `uv` is unavailable).
- **JAX** (with `optax`, `flax` or plain pytrees) for the brain model, sim, and training. Vectorize with `vmap`, compile with `jit`.
- **Training compute:** UF HiPerGator (SLURM). Provide `sbatch` scripts. Also runs on a single local GPU at reduced scale.
- **Data:** `pandas`, `pyarrow` (Parquet), `neuprint-python` for MaleCNS, FlyWire Codex downloads.
- **Server:** FastAPI plus WebSockets (`uvicorn`).
- **Front end:** React plus Vite plus TypeScript, `three` via `@react-three/fiber` and `@react-three/drei`, Zustand for state, Tailwind for UI.
- **Audio analysis (stretch):** `librosa`.
- **LLM:** Google Gemini API (server side only).
- **Quality:** `ruff`, `mypy`, `pytest`, ESLint, Prettier.
- **Deploy:** Docker, Vultr.

---

## 4. Repository layout

The proposed tree below is retained from the original spec. For the current layout and commands, see the [repository guide](../../development/REPOSITORY.md).


```
fly_saber/
  README.md
  Makefile                  # setup, data, train, serve, web, test, lint
  pyproject.toml
  configs/                  # YAML configs, loaded into dataclasses
    subgraph.yaml
    model.yaml
    runner.yaml
    saber.yaml
    train_bc.yaml
    train_ppo.yaml
    variants.yaml
  data/                     # gitignored; raw and processed data
    raw/
    processed/
    maps/
  docs/
    ARCHITECTURE.md
    DATA_NOTES.md           # what the real files contain, deviations from this spec
    PROTOCOL.md             # WebSocket message spec
    RESULTS.md
  fly_saber/
    connectome/
      load_malecns.py
      load_flywire.py
      subgraph.py           # cell-type based extraction
      signs.py              # neurotransmitter to sign
      variants.py           # rewire, random, sign-shuffle
      metadata.py           # neuron info for Brain Explorer
    model/
      brain.py              # connectome rate network
      mlp.py                # matched-parameter baseline
      interventions.py      # gains, opto, mood, eye masks
    sim/
      eye.py                # ommatidia lattice and analytic rendering
      runner.py             # Fly Runner env
      saber.py              # Fly Saber env
      experts.py            # scripted perfect players
    maps/
      beatsaver_client.py
      parse_v2.py
      parse_v3.py
      normalize.py
      adapt.py
      autogen.py            # librosa beat-based map generation (stretch)
    train/
      bc.py
      ppo.py
      evaluate.py
      checkpoints.py
    server/
      app.py
      session.py
      protocol.py
      narrator.py           # Gemini
      replay.py
  scripts/
    fetch_data.py
    build_subgraph.py
    build_variants.py
    fetch_maps.py
    record_replays.py
    slurm/
      train_bc.sbatch
      train_ppo.sbatch
  tests/
  web/
    src/
      game/                 # Three.js scenes for each game
      brain/                # neuron cloud, explorer
      controls/             # lesion, opto, mood, eye panels
      eyeview/              # hexagonal fly's-eye view
      modes/                # arcade, versus, you-vs-fly
      net/                  # websocket client, replay loader
  docker/
    Dockerfile.server
    Dockerfile.web
```

---

## 5. Connectome pipeline (`fly_saber/connectome/`)

### 5.1 Sources
- **MaleCNS v1.0** (primary): Janelia neuPrint (`neuprint-python`, requires a free token). Also check for bulk downloads (Parquet or CSV) on the Janelia / Google release pages, which are faster than querying.
- **FlyWire** (female brain, for Versus Mode): FlyWire Codex bulk downloads (connections, neurotransmitter predictions, cell types, classification, soma coordinates).

Inspect what each source actually provides and record it in `docs/reference/DATA_NOTES.md`. Required per neuron: id, cell type, coarser class or superclass, predicted neurotransmitter, 3D soma (or centroid) position, side (left/right). Required per edge: pre id, post id, synapse count.

### 5.2 Subgraph extraction
Extract the visual-to-motor pathway, target **5,000 to 20,000 neurons** (configurable in `subgraph.yaml`):

1. **Seed sets by cell type:** photoreceptors (R1 to R6, R7, R8), lamina (L1 to L5), medulla (Mi, Tm, TmY, T4, T5 and others), lobula and lobula plate (LC types including LPLC2, lobula plate tangential cells such as HS and VS), and descending neurons (DN types, including the giant fiber DNp01).
2. Include intermediate neurons on short paths (configurable hop limit) between visual seeds and DNs, ranked by synapse weight.
3. Drop edges below a synapse-count threshold (start with 5, configurable).
4. If over budget, prune lowest-weight intermediate neurons first; never drop seed cell types needed for lesion demos (T4, T5, LPLC2, LC types, DNs).
5. Use both eyes and both hemispheres (left/right dodging needs both).

Output `data/processed/{dataset}_subgraph/`: `neurons.parquet`, `edges.parquet`, `meta.json` (counts, thresholds, source version).

### 5.3 Signs
Map neurotransmitter to sign in `signs.py`: acetylcholine `+1`, GABA `-1`, glutamate `-1` (inhibitory in the fly central brain via GluCl; note this simplification in docs), others (dopamine, serotonin, octopamine) `0` for fast synaptic transmission but tagged as modulatory groups for the mood sliders. Make the mapping configurable.

### 5.4 Neuron groups
Build `groups.json`: named groups used by the UI and interventions, for example `T4`, `T5`, `LPLC2`, `LC_all`, `lamina`, `photoreceptors_left`, `photoreceptors_right`, `DN_all`, `giant_fiber`, `modulatory_dopamine`, `modulatory_octopamine`. Each group is a list of subgraph indices.

### 5.5 Brain Explorer metadata
`metadata.py` produces `explorer.json`: per neuron (index, id, cell type, class, side, position, NT, top inputs and outputs by type) and per cell type a short human-written description of known biological function. Write descriptions for at least 15 key types (T4, T5, L1, L2, Mi1, Tm3, LPLC2, LC4, LC11, HS, VS, giant fiber, R1 to R6, R7, R8). Keep descriptions short and factual; if unsure, say "function not well characterized."

### 5.6 Variants (`variants.py`)
Each variant is a new edge list over the same neurons:
- `real`: unchanged.
- `rewired`: degree-preserving shuffle via repeated double-edge swaps preserving every neuron's in-degree and out-degree. Sign stays attached to the presynaptic neuron (Dale's law preserved). Synapse weights travel with edges.
- `random_sparse`: Erdos-Renyi graph with the same neuron count and edge density, signs drawn with the same overall ratio.
- `sign_shuffled`: real topology, neuron signs permuted.
- `mlp`: not a graph; a dense MLP baseline with matched trainable parameter count (`model/mlp.py`).

Tests: in/out degrees identical after rewiring; edge count identical; seeded and deterministic.

---

## 6. Brain model (`fly_saber/model/brain.py`)

### 6.1 Dynamics
Leaky rate network, Euler-integrated, `K` substeps per game frame (default `K=4`, configurable):

```
tau_i * dv_i/dt = -v_i + sum_j W_ij * s_j * g_j * f(v_j) + b_i + I_i(t) + opto_i
r_i = gain_i * f(v_i)        # f = relu (or softplus), configurable
```

- `W_ij`: fixed sparsity from the subgraph, initialized from synapse counts (log-scaled and normalized). Edge-level weights are fixed by default.
- `s_j`: presynaptic sign (Section 5.3).
- **Trainable per-neuron parameters:** `log_tau_i`, `b_i`, `g_i` (synaptic output scale), plus a trainable scale per (pre cell type, post cell type) pair. This is the key difference from viral demos that trained only a readout.
- `I_i(t)`: visual input to photoreceptors only.
- Implement the sparse matmul with an edge list and `jax.ops.segment_sum` (reliable under `jit` and `vmap`). Benchmark early; fall back to dense only if the subgraph is small enough.

### 6.2 Readout
Action logits come from a small linear layer on descending neuron rates only (`DN_all` group). Value head for PPO also reads DNs. Keep readout parameters a small fraction of total trainable parameters and report the ratio in the README.

### 6.3 State
Recurrent state `v` carries over between frames. Reset at episode start. Return all rates per frame for visualization.

### 6.4 Interventions (`interventions.py`)
All applied at inference time, no retraining:
- **Lesion:** set `gain_i = 0` and block output for neurons in selected groups.
- **Optogenetics:** add a large positive constant to `opto_i` for selected groups.
- **Mood sliders:** each slider is a named, documented mapping, for example `arousal` scales DN input gain and adds input noise; `reward_sensitivity` scales the dopaminergic group gains. Define them in config with clear descriptions; validate each one changes behavior in the predicted direction (`evaluate.py`) and put the validation plot in `docs/results/README.md`.
- **Eye controls:** masks and transforms on the ommatidia input (Section 7.1).

Interventions are a single pytree passed to the step function so the server can update them live.

---

## 7. Simulation (`fly_saber/sim/`)

### 7.1 Fly eye (`eye.py`)
- Two hexagonal ommatidia lattices (left and right eye), configurable count per eye (default about 700) and wide field of view.
- **Analytic rendering, no 3D graphics in training:** objects are boxes or spheres with positions and sizes. For each ommatidium, compute luminance from objects using angular position and a Gaussian acceptance angle (about 5 degrees, configurable). Add a ground plane texture and lane markings so optic flow exists. Fully vectorized in JAX.
- **Mapping ommatidia to photoreceptor neurons:** use column or retinotopic assignments if the data provides them; otherwise project photoreceptor 3D positions onto a sphere per eye and match to lattice points by nearest angle. Document the method in `DATA_NOTES.md`.
- **Eye controls:** `blind_left`, `blind_right`, `blur` (wider acceptance angle), `facet_dropout`, `noise`.
- Expose the per-frame ommatidia image for the fly's-eye view.

### 7.2 Fly Saber (`saber.py`), the main game
**Layout and timing**
- Notes approach the fly in a 4 (lanes) x 3 (layers) grid along the forward axis, spawned from a normalized map (Section 8) so each note arrives at the hit plane exactly at its `time_sec`.
- Note jump speed and spawn distance come from map metadata when present, else config defaults. Expose a global speed multiplier for curriculum.
- Sim time advances in fixed steps (default 30 Hz, matching the server). Audio sync is a front end concern (Section 14): the sim clock is the source of truth.

**What the fly sees**
- Blocks are rendered into the ommatidia analytically (Section 7.1), so an approaching block expands across the eye as a looming stimulus.
- **Cut direction must be resolvable by low-resolution fly vision.** Never encode it as a small arrow. Default encoding: direction as distinct block brightness patterns or shapes large enough to span several ommatidia (for example a bright half on the side the cut should start from). Color (red/blue saber) is encoded as a separate brightness channel or mapped to which saber must hit it. Keep the encoding configurable and document it in the README.
- Bombs have a distinct shape (sphere) and must not be hit. Walls (obstacles) are large boxes the fly's head position must avoid (stretch; may be dropped from maps in `adapt.py`).

**Actions (factorized, discrete)**
- Two sabers (red and blue), matching Beat Saber's two colors. Default simplification for must-ship: **one saber per color, each with its own position head (12 cells plus `idle`) and direction head (4 directions plus `no_cut`).** If two sabers are too hard to train in time, fall back to a single saber that hits both colors and document it.
- Directions: 4 (up, down, left, right) for must-ship; 8 including diagonals is stretch. `adapt.py` maps map diagonals to the nearest supported direction when running with 4. "Any direction" notes accept any cut.

**Scoring**
- A note is hit if the correct-color saber is in the note's cell with the correct direction within a timing window (default plus or minus 100 ms, configurable) of `time_sec`.
- Reward: `+1` hit, `-0.5` miss, `-0.25` wrong direction or wrong color, `-1` bomb hit. Small penalty for cutting when no note is present to discourage spamming.
- Track combo, accuracy, and a 0 to 100 score for the UI.

**Episodes and curriculum**
- One episode is one song (or a fixed-length window of a song during training, for shorter rollouts and more variety).
- Curriculum by map difficulty score (Section 8.3): start with low-density maps and position-only scoring (direction ignored), then enable direction, then raise density and speed.
- Train/held-out split of maps by song, so held-out evaluation is on songs never seen in training.

**Implementation**
- Pure-functional JAX env (`reset(key, map_id)`, `step(state, action)`), `vmap`-able, gymnax-style. Maps are preloaded as padded arrays so many songs run in parallel in one batch.

### 7.3 Fly Runner (`runner.py`), stretch
- 3 lanes, forward speed increasing over time.
- Obstacle types: low barrier (jump), high barrier (duck), blocking wall (change lane).
- Actions (discrete, 5): `noop`, `left`, `right`, `jump`, `duck`.
- Reward: `+0.01` per surviving step, `+1` per obstacle passed, `-1` and episode end on collision.
- Procedurally generated, seeded. Same env interface as Fly Saber.
- Purpose: cross-game lesion demo (for example looming detectors matter for both games, direction-related cells mostly for Fly Saber).

### 7.4 Experts (`experts.py`)
Scripted perfect players reading the true future state. For Fly Saber, the expert reads the normalized map, moves each saber to the next note's cell ahead of time, and cuts in the right direction at `time_sec`. It must score near 100 on every demo map; use it to validate parsers and timing before training. Used for behavior cloning and as an upper-bound baseline.

### 7.5 Shared state for rendering
Each env exposes a `render_state(state)` returning plain object positions, types, score, and time. The server streams this to Three.js, which renders the same game beautifully. Training never touches Three.js.

---

## 8. Song map pipeline (`fly_saber/maps/`)

**Priority:** parsing, normalizing, and adapting maps (8.2, 8.3) plus the demo allowlist (8.4) are **must ship**, because Fly Saber cannot run without them. Start in hour 0, in parallel with everything else. The automated BeatSaver scraper for a large training set (8.1 at scale) is target tier; for must-ship, a handful of manually downloaded maps is enough.

### 8.1 Fetch (`beatsaver_client.py`, `scripts/data/fetch_maps.py`)
- BeatSaver public API (`api.beatsaver.com`). Read its docs; respect rate limits; cache everything under `data/maps/`. Never call the API at demo time.
- Filters: rating, difficulty availability (Easy to Hard), BPM range, and license or song source where available.
- Download zip, extract `Info.dat`, difficulty `.dat` files, audio, cover.

### 8.2 Parse and normalize
- Detect map format version per difficulty file. Implement `parse_v2.py` and `parse_v3.py` (v4 optional). Inspect real files before writing parsers.
- Normalize to one schema saved as JSON or Parquet:
  `{time_sec, lane (0-3), layer (0-2), color, cut_direction, is_bomb}` plus walls `{time_sec, duration_sec, lane, width, ...}` and song metadata.
- Convert beats to seconds with BPM, handling BPM-change events.
- Tests with small fixture maps for each version.

### 8.3 Adapt for the fly (`adapt.py`)
- Map cut direction to color or shape encoding.
- Optional layer merging and note density cap.
- Difficulty score: notes per second, direction changes, lane changes.

### 8.4 Copyright rules (mandatory)
- Training uses note charts only; audio is not needed for training.
- The public demo and deployed app only include maps whose audio is royalty-free or Creative Commons, or maps generated from such audio. Maintain `data/maps/demo_allowlist.json` with source and license per map.
- Credit mappers in the UI for every displayed map.
- **Stretch, "upload a song":** `autogen.py` uses `librosa` beat and onset detection on user-uploaded audio to generate a playable map. Uploaded audio is processed in memory and not stored or redistributed.

### 8.5 Evaluation
Hold out a set of maps never used in training. Report hit rate on train vs. held-out maps per brain variant in `docs/results/README.md`.

---

## 9. Gemini brain narrator (`src/server/narrator.py`)

- **Explain:** after a crash or miss, the server builds a compact event summary (what happened, active interventions, top-activated cell types in the last second, which groups were lesioned) and asks Gemini for a 1 to 2 sentence plain-English explanation. Display in the UI. Prompt must instruct the model to describe only what the summary contains and not invent biology.
- **Command:** user types something like "make the fly more cautious" or "turn off its motion vision"; Gemini returns structured JSON (use structured output or function calling) matching the interventions schema; server validates against the known groups and slider ranges before applying.
- API key in env var, server side only. Rate-limit and cache. The app must work fully if Gemini is unavailable.

---

## 10. Optional: Tiger Data

Only if replays or the leaderboard are built. Store sessions, scores, and per-cell-type activity time series in Tiger Data (PostgreSQL based). Use a continuous aggregate for leaderboard and activity dashboards. Keep it behind an interface so the app runs without it.

---

## 11. Optional: physical control box

Arduino or Raspberry Pi with toggle switches (lesion groups) and potentiometers (mood sliders). Sends JSON lines over serial or WebSocket to the server, which converts them into the same control messages as the web UI. Only after hour 14 with a spare person.

---

## 12. Training (`fly_saber/train/`)

1. **Behavior cloning (`bc.py`):** roll out experts in thousands of parallel envs, train the brain to imitate actions (cross-entropy). Optionally a few rounds of DAgger. This gives a working agent early.
2. **PPO fine-tuning (`ppo.py`):** PureJaxRL-style single-file PPO, fully jitted, thousands of envs, recurrent state handled per env. Start from the BC checkpoint.
3. **Variants:** run the exact same configs for every variant in `variants.yaml`. Multiple seeds per variant (target 5, stretch 10 or more).
4. **Logging:** metrics to JSON or CSV plus optional Weights & Biases. Save checkpoints with config hash.
5. **SLURM:** `infra/slurm/*.sbatch` for HiPerGator with GPU requests and array jobs over seeds and variants.

### Evaluate (`evaluate.py`)
- Learning curves per variant (mean and confidence interval across seeds).
- Final score, sample efficiency (steps to threshold).
- Lesion matrix: score drop per group per game.
- Mood slider and knob validation.
- Held-out map performance (Fly Saber).
- Write plots to `docs/figures/` and a summary to `docs/results/README.md`. Report honestly whatever the result is; "wiring did not help" is a valid finding.

---

## 13. Server (`fly_saber/server/`)

- FastAPI app. `GET /api/brains`, `GET /api/games`, `GET /api/maps`, `GET /api/explorer/{brain}`, `WS /ws/session`.
- A session holds env state, brain state, interventions, and steps at a fixed rate (target 30 Hz) using jitted functions on CPU or GPU.
- **Client to server messages:** `start {game, brain, map_id, seed}`, `set_interventions {...}`, `human_action {...}` (You vs. the Fly), `pause`, `reset`, `narrate`.
- **Server to client messages:** `frame {render_state, score, t}`, `activity` (binary: uint8-quantized rates for all subgraph neurons, at 15 Hz, plus per-cell-type means at 30 Hz), `eye` (ommatidia image), `event {crash|miss|hit}`, `narration {...}`.
- Versus Mode runs two brains on identically seeded envs in one session.
- Spec everything in `docs/reference/PROTOCOL.md`.

### Replay fallback (`replay.py`, `scripts/training/record_replays.py`)
Record full sessions (frames, activity, eye images, events) for every game and brain variant plus key lesions to compressed files. The front end plays them with no server. Build this in Milestone 4, before polish.

---

## 14. Front end (`web/`)

- **Layout:** game view (left), 3D brain (right), control panel (bottom or side), fly's-eye view (inset).
- **3D brain:** instanced points at real neuron 3D positions, colored by cell type, brightness from live activity. Orbit controls. Click to select a neuron or cell type.
- **Brain Explorer panel:** identity, description, connections, live activity sparkline.
- **Control panel:** lesion toggles per group, opto buttons, mood sliders, eye controls, reset brain.
- **Fly's-eye view:** hexagonal mosaic renderer of both eyes.
- **Fly Saber scene (priority):** Three.js scene with a track, approaching blocks (direction patterns matching the fly's encoding, so humans see what the fly sees), two glowing sabers driven by the fly's actions, hit and miss effects, combo and score HUD. Original art and name only.
- **Song playback:** play allowlisted audio with the Web Audio API and position blocks from `AudioContext.currentTime` so visuals never drift from the music. The server sim and the audio clock are synced at song start and resynced periodically.
- **Map picker:** browse demo maps (title, mapper credit, difficulty, license), plus an "unseen map" badge for held-out songs.
- **Fly Runner scene:** stretch only.
- **Modes:** Arcade, Versus (split screen), You vs. the Fly (keyboard), Replay.
- **Narrator panel:** explanation text and a command box.
- Must run at 30 fps on a laptop. Mobile layout is stretch.

---

## 15. Deployment

- Dockerize server and web. Deploy on **Vultr** (GPU instance only if CPU inference cannot hold 30 Hz; benchmark first).
- Serve the web app and replays as static files; WebSocket to the server.
- Domain registered through **GoDaddy Registry**, HTTPS via Caddy or similar.
- Document deployment in the README.

---

## 16. README requirements

Title, one-paragraph pitch, demo GIF, live URL, features, architecture diagram, how the brain model works (what is trained, parameter counts, readout ratio), data sources and credits (MaleCNS, FlyWire, BeatSaver mappers, music licenses), results (learning curves, real vs. scrambled, lesion matrix, held-out maps), setup and run instructions, deployment (Vultr), prize integrations, limitations and honest caveats (rate model simplification, sign simplifications, ommatidia mapping method, this is not a living fly).

---

## 17. Milestones (24-hour plan, 4 people)

Roles: **A** connectome and model, **B** sim and training, **C** server, maps, and integrations, **D** front end.

| Hours | Milestone | Done when |
|---|---|---|
| 0 to 4 | **M1 Foundations** | Subgraph built with groups and signs (A). Map parsers (v2, v3), normalizer, adapter, and 3 to 5 allowlisted demo maps ready (C). Fly Saber env plus scripted expert running vectorized on those maps, expert scoring near 100 (B). Three.js Fly Saber scene renders a map from a JSON file with audio in sync (D). |
| 4 to 8 | **M2 Brain plays** | Brain forward pass benchmarked, rewired variant built (A). BC on easy maps with position-only scoring produces a fly that hits clearly more notes than random (B). Server streams a live Fly Saber session from a real checkpoint (C). 3D neuron cloud shows live activity (D). |
| 8 to 14 | **M3 Must-ship complete** | Direction scoring enabled, PPO fine-tuning, all variants with multiple seeds on HiPerGator (B). Lesion and interventions wired end to end (A, C). Versus Mode real vs. scrambled on the same song (D). BeatSaver scraper building a larger training set with a held-out split (C). **Hour 14 tier decision.** |
| 14 to 18 | **M4 Target features and safety net** | Replay recording for every demo map and variant (C). Curriculum training on the large map set, held-out map evaluation (B). Brain Explorer, opto, mood sliders, fly's-eye view, eye controls (D, A). Deployed to Vultr with domain (C). |
| 18 to 22 | **M5 Polish and stretch** | Gemini narrator (C). Evaluation figures and docs/results/README.md (A, B). UI polish, You vs. the Fly, upload-a-song (D, C). Fly Runner only if everything else is done. |
| 22 to 24 | **M6 Ship** | README complete, Devpost submission with public repo link, demo rehearsed, replay fallback verified offline. |

---

## 18. Acceptance tests (must pass before submission)

- `make test` passes; `ruff` and `mypy` clean.
- Rewired graph preserves in and out degree for every neuron.
- Map parsers produce identical normalized output for equivalent v2 and v3 fixtures.
- Scripted expert scores near 100 on every demo map (validates parsing and timing).
- Trained real-wiring fly beats a random policy on Fly Saber hit rate by a clear margin across seeds, on both training and held-out maps.
- Lesioning looming detectors (LPLC2 and related LC types) measurably reduces Fly Saber hit rate.
- Song audio stays in sync with notes in the browser (no visible drift over a full song).
- Live session holds 30 Hz with activity streaming on the deployed server.
- Replay mode works with the server off.
- Public deployment contains only allowlisted audio.
- README results section is filled with real numbers.

---

## 19. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Connectome wrangling eats hours | Pre-download data; inspect formats before the event; strict neuron budget. |
| Sparse JAX ops too slow | Benchmark in M2; reduce neuron budget or substeps; dense fallback for small graphs. |
| Fly fails to learn Fly Saber | BC first gives an early working agent; curriculum from position-only scoring on sparse maps; single saber fallback; 4 directions only; larger direction encodings. |
| Fly cannot resolve cut direction | Make direction patterns larger and higher contrast; verify by checking ommatidia images in the fly's-eye view; if still failing, ship position plus color scoring and say so honestly. |
| Audio drift in browser | Drive note positions from the Web Audio clock (`AudioContext.currentTime`), not frame counts. |
| Real wiring does not beat scrambled | Report honestly; the interactive demo works regardless. |
| HiPerGator queue delays | Request allocation early; must-ship tier trains on one GPU. |
| Server crash at demo | Replay fallback; rehearse offline. |
| Copyright on songs | Allowlist only royalty-free or CC audio; train on charts only. |
| Scope creep | Hour 14 tier decision; must-ship always works on `main`. |

---

## 20. Demo script (3 minutes)

1. Fly Saber playing a song, 3D brain lighting up beside it.
2. "The viral Beat Saber fly replayed movements. Ours actually sees the blocks." Show the fly's-eye view of a block looming in.
3. Load a map the fly has never seen; it plays it cold.
4. Judge flips a lesion toggle on looming detectors; the fly starts missing.
5. Max arousal, the fly swings wildly; optogenetically trigger escape mid-song.
6. Real vs. scrambled side by side on the same song.
7. Narrator explains a miss in plain English.
8. "It's live at [domain]. Go break a fly brain."
