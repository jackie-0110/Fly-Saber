# FlySaber: 3-Person Work Split

Companion to `FLYSABER_HANDOFF.md`. That file is the full spec; this one says who builds what.

**Principle:** every person owns one piece of the AI/ML work plus one piece of the engineering. Nobody is "just front end" or "just infra," everyone can explain the ML to judges, and training waits never leave someone idle.

---

## Roles at a glance

| | Person 1: Brain and RL | Person 2: Game, Imitation, and Server | Person 3: Experience and Analysis |
|---|---|---|---|
| **AI/ML** | Connectome network model, variants, PPO fine-tuning | Fly eye input model, scripted expert, behavior cloning, curriculum, held-out generalization | Evaluation suite, lesion and interpretability analysis, Gemini narrator |
| **Engineering** | Connectome data pipeline, HiPerGator jobs | Map pipeline, Fly Saber env, server, deploy | Front end (Three.js game, 3D brain, control panel), replay player |
| **README sections** | Brain model, variants, training | Data pipeline, game, maps, deployment | Results, figures, features, demo GIF |
| **Pitch role** | "How the fly brain works" | "How it learned to play" | Runs the live demo |

---

## Person 1: Brain and RL

### AI/ML
- Brain model in JAX (`model/brain.py`): leaky rate dynamics, trainable per-neuron parameters, cell-type pair scales, DN readout.
- Interventions at the model level (`model/interventions.py`): lesion, opto, mood sliders, eye masks as one pytree.
- Variants (`connectome/variants.py`): degree-preserving rewire, random sparse, sign-shuffled, matched MLP baseline. Tests for degree preservation.
- PPO fine-tuning (`train/ppo.py`): PureJaxRL-style, recurrent state per env, starts from Person 2's BC checkpoint.
- Multi-seed runs for every variant.

### Engineering
- Connectome download and inspection, `docs/DATA_NOTES.md`.
- Subgraph extraction, signs, `groups.json`, `explorer.json` metadata and cell-type descriptions.
- HiPerGator SLURM scripts and job management.

### Delivers
- Hour 4: `groups.json` and interventions schema (to Persons 2 and 3).
- Hour 6 to 8: brain forward pass that Person 2's BC code can call.
- Hour 14: PPO checkpoints for all variants (to Person 2 for the server, Person 3 for evals).

---

## Person 2: Game, Imitation, and Server

### AI/ML
- Fly eye rendering (`sim/eye.py`): ommatidia lattice, analytic looming rendering, photoreceptor mapping, eye controls. This is the model's input representation, so it is real ML design work.
- Scripted expert (`sim/experts.py`), validated near 100 on every demo map.
- Behavior cloning (`train/bc.py`), optional DAgger.
- Curriculum design: position-only scoring first, then direction, then density and speed.
- Train/held-out map split and held-out evaluation ("plays songs it has never heard").

### Engineering
- Map pipeline (`maps/`): v2 and v3 parsers, normalizer, adapter, demo allowlist, BeatSaver scraper.
- Fly Saber env (`sim/saber.py`).
- FastAPI and WebSocket server, live sessions, Versus backend.
- Vultr deploy and domain.

### Delivers
- Hour 2: normalized map schema and 3 demo maps as JSON (to Person 3, so the game scene can start).
- Hour 4: `render_state` format and a runnable env plus expert (to Person 1).
- Hour 8: server streaming a live session (to Person 3).
- Hour 18: public URL.

---

## Person 3: Experience and Analysis

### AI/ML
- Evaluation suite (`train/evaluate.py`): learning curves with confidence intervals across seeds, real vs. scrambled comparison, sample efficiency.
- Lesion matrix: score drop per neuron group.
- Mood slider validation: confirm each slider shifts behavior in the predicted direction.
- Interpretability probes: which cell types respond to approaching blocks, to block position, and to cut direction. These feed the Brain Explorer ("this neuron fires for left-cut blocks").
- Gemini brain narrator (`server/narrator.py`): explain-a-miss and natural-language commands with validated structured output.

### Engineering
- Three.js Fly Saber scene, sabers, effects, HUD.
- Audio synced to the Web Audio clock, map picker with mapper credits.
- 3D neuron cloud, Brain Explorer panel, control panel, fly's-eye view, Versus split screen.
- Replay recording script and replay player (the demo safety net).

### Delivers
- Hour 4: game scene playing a map JSON with synced audio.
- Hour 12: lesion panel working live.
- Hour 18: figures and `docs/RESULTS.md` draft.
- Hour 20: demo GIF.

---

## Hour 0 contracts (30 minutes, all three)

Write these into `docs/PROTOCOL.md` and `docs/DATA_NOTES.md` before real code, so everyone builds against fake data in parallel:

| Contract | Owner | Consumers |
|---|---|---|
| Normalized map schema | Person 2 | 1, 3 |
| `render_state` format | Person 2 | 3 |
| WebSocket protocol (both directions) | Person 2 | 3 |
| `groups.json` and interventions schema | Person 1 | 2, 3 |
| Brain step function signature and checkpoint format | Person 1 | 2, 3 |
| Activity stream format (quantized rates, cell-type means) | Person 1 | 3 |

---

## Timeline

| Hours | Person 1 | Person 2 | Person 3 |
|---|---|---|---|
| 0 to 0.5 | Contracts (all) | Contracts (all) | Contracts (all) |
| 0.5 to 4 | Subgraph, signs, groups | Map parsers, 3 demo maps, Fly Saber env | Game scene with synced audio on a map JSON |
| 4 to 8 | Brain forward pass, benchmark, rewired variant | Eye rendering, expert near 100, BC on easy maps, server streaming | 3D neuron cloud on fake activity, WebSocket client, evaluation harness skeleton |
| 8 to 14 | PPO from BC checkpoint, all variants, multi-seed on HiPerGator | Direction scoring in curriculum, interventions over WebSocket, Versus backend, BeatSaver scraper | Lesion panel live, Versus split screen, first learning-curve plots |
| **14** | **Tier decision (all)** | | |
| 14 to 18 | Monitor runs, write Brain Explorer descriptions, help Person 3 with probes | Large-map curriculum training, held-out eval, deploy and domain | Lesion matrix, slider validation, interpretability probes, Brain Explorer, opto, sliders, fly's-eye view |
| 18 to 22 | README brain and training sections | Gemini narrator handoff support, bug fixes, README data and deploy sections | Replay recording and player, Gemini narrator, `RESULTS.md`, UI polish, demo GIF |
| 22 to 24 | Pitch rehearsal, Devpost, offline replay check (all) | | |

---

## Handoff points

- **Hour 2:** Person 2 gives Person 3 demo maps as JSON. Game scene starts on real data.
- **Hour 4:** Person 2 gives Person 1 the env plus expert. Person 1 gives everyone `groups.json`.
- **Hour 8:** Person 1's brain plus Person 2's BC produce the first checkpoint. Person 2's server streams it to Person 3's front end. **This is the first end-to-end moment; celebrate and commit.**
- **Hour 14:** PPO checkpoints land. Person 3 starts real evaluations.

## Filling training waits

PPO and curriculum runs take time on HiPerGator. While jobs run:
- Person 1 pairs with Person 3 on interpretability probes (Person 1 knows the cell types).
- Person 2 pairs with Person 3 on replay recording (Person 2 knows the session format).
- Nobody blocks on a training run; always have the previous checkpoint wired in.

## Shared rules

- `main` always runs the must-ship demo. Work on branches, merge small and often.
- Every person writes tests for their own core logic (parsers, rewiring, env determinism, protocol).
- Run `make lint test` before merging.
- Check in every 4 hours (2 minutes each): done, blocked, next.
- No em dashes in user-facing text or docs.

## Cut for 3 people

Fly Runner, Fly Flight, Fruit Fly Ninja, Tiger Data, hardware control box, leaderboard, Brain Lab. FlyWire vs. MaleCNS only if Person 1 finishes early. Gemini narrator is the first thing to cut after hour 14 if behind.
