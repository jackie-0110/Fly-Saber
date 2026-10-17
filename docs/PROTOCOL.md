# Contracts and WebSocket Protocol

Fill these in during the hour 0 contracts session (30 minutes, all three). Everyone builds against these with fake data until the real pieces land. Change a contract only after telling its consumers.

| Contract | Owner | Consumers | Status |
|---|---|---|---|
| Normalized map schema | Person 2 | 1, 3 | Draft |
| `render_state` format | Person 2 | 3 | Draft |
| WebSocket protocol (both directions) | Person 2 | 3 | Draft |
| `groups.json` and interventions schema | Person 1 | 2, 3 | groups done, interventions TODO |
| Brain step function signature and checkpoint format | Person 1 | 2, 3 | Done |
| Activity stream format (quantized rates, cell-type means) | Person 1 | 3 | TODO |

## 1. Normalized map schema (Person 2)

One JSON file per playable (song, difficulty): `data/maps/normalized/{map_id}.json`. `data/maps/` is gitignored, so other people get maps from the REST endpoints in section 3, not from git. `map_id` is `{beatsaver_key}_{difficulty}` in lowercase, for example `3a2b1_hard`.

```jsonc
{
  "schema_version": 1,
  "map_id": "3a2b1_hard",
  "split": "train",                  // "train" | "heldout"; heldout gets the "unseen map" badge
  "song": {
    "title": "Example Song",
    "artist": "Example Artist",
    "mapper": "example_mapper",      // credit in the UI for every displayed map
    "bpm": 128.0,                    // initial BPM, informational only
    "duration_sec": 184.2,
    "audio_url": "/api/maps/3a2b1_hard/audio",  // null when the audio is not allowlisted
    "license": "CC BY 4.0",          // null when not allowlisted
    "attribution": "\"Example Song\" by Example Artist (https://...), licensed under CC BY 4.0",  // show with the song; null when not allowlisted
    "source_url": "https://beatsaver.com/maps/3a2b1"
  },
  "difficulty": {
    "name": "Hard",                  // Easy | Normal | Hard | Expert | ExpertPlus
    "note_speed_mps": 10.0,          // meters per second toward the player
    "spawn_distance_m": 12.0         // a note becomes visible this far before the hit plane
  },
  "notes": [
    { "id": 0, "time_sec": 2.344, "lane": 1, "layer": 0, "color": 0, "cut_direction": 1, "is_bomb": false },
    { "id": 1, "time_sec": 2.812, "lane": 2, "layer": 1, "color": null, "cut_direction": null, "is_bomb": true }
  ],
  "walls": [
    { "time_sec": 10.0, "duration_sec": 1.5, "lane": 0, "layer": 0, "width": 1, "height": 3 }
  ]
}
```

Rules:

- `time_sec` is seconds from the start of the audio file, with BPM changes and any song offset already applied. It is the moment the note crosses the hit plane.
- `notes` is sorted by `time_sec` and `id` is its index in that array. Frames and events refer to notes by `id`.
- `lane` 0 to 3, left to right from the player's view. `layer` 0 to 2, bottom to top. Grid cell index is `layer * 4 + lane` (0 to 11).
- `color`: 0 red (left saber), 1 blue (right saber), `null` for bombs.
- `cut_direction` keeps the original 9-value numbering: 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right, 6 down-left, 7 down-right, 8 any. `null` for bombs. Reducing diagonals to 4 directions happens in `adapt.py` for the sim; the front end draws whatever is in the file.
- A note's distance from the hit plane at song time `t` is `(time_sec - t) * note_speed_mps`. Both the sim and the Three.js scene use this formula.
- `walls` may be empty (walls are stretch).

## 2. `render_state` (Person 2)

The front end already has the full map from REST and positions notes from the audio clock (spec 14), so `render_state` carries only what the map cannot tell it: what the fly is doing and the score. It is the payload of the `frame` message.

```jsonc
{
  "t": 12.533,                       // sim song time in seconds, same clock as note time_sec
  "step": 376,                       // sim step index, 30 Hz
  "sabers": [
    { "color": 0, "cell": 5, "cut": null },   // cell 0-11 or null (idle, saber lowered)
    { "color": 1, "cell": 6, "cut": 3 }       // cut 0 up, 1 down, 2 left, 3 right, or null
  ],
  "score": {
    "score": 87,                     // 0 to 100, int
    "combo": 14,
    "max_combo": 31,
    "hits": 52,
    "misses": 6,
    "accuracy": 0.897                // hits / notes judged so far, 0 to 1
  }
}
```

### Env interface for the brain (Person 2 to Person 1)

`fly_saber.env.SaberEnv`, gymnax style, every method pure and `jit`/`vmap` safe:

- `reset(notes, speed_mps, start_sec, end_sec) -> (obs, state)`; `notes` from `fly_saber.maps.adapt.note_arrays`.
- `step(state, action) -> (obs, state, reward, done, events)`.
- `obs`: float32 `(2, n_eye)`, values 0 to 1. Channel 0 is luminance, for R1 to R6. Channel 1 is color (red notes high, blue low, background 0.5), for R7 and R8.
- Eye for training and live play: `training.brain_policy.build(model_cfg, saber_cfg, eye_cfg, key)` makes the env see through the brain's own eye columns (angles from `brain.connectome.metadata.column_angles`, the same as `columns.json`), so `n_eye` is the 437 columns in `load_graph`'s column-table order and `obs` is exactly the brain's `eye` argument. `BrainPolicy` wraps `make_step` in the `Policy` interface of `training/bc.py`. Without a brain, `SaberEnv.create` falls back to a synthetic hex lattice (`fly_saber.eye.lattice`, left eye first).
- `action`: int32 `(n_sabers, 2)`, rows red then blue: `[cell, cut]`, cell 0 to 11 or 12 (idle), cut 0 up, 1 down, 2 left, 3 right, or 4 (no cut). That is a 13-way position head and a 5-way direction head per saber. `n_sabers` is 1 when `saber.yaml` sets `two_sabers: false`.
- `fly_saber.experts.saber_expert(env, state)` returns the perfect action, for behavior cloning.
- Eye controls (spec 7.1) live in `state.eye` (`fly_saber.eye.EyeControls`) and change between frames with no recompile: `fly_saber.env.set_eye_controls(state, blind_left=1.0)` returns the new state. Fields: `blind_left`, `blind_right` (0 to 1, contrast that eye loses), `blur_deg` (added to the acceptance FWHM), `facet_dropout` (0 to 1, fraction of columns, the same columns for a given `seed`), `noise` (luminance std, fresh each frame), `seed`. Lost columns read as background 0.5. Which eye a column belongs to comes from the brain's column `side` (`SaberEnv.left_eye`), not from the sign of its azimuth, because the eyes overlap past the midline.

A note leaves the scene when an `event` for its `id` arrives (hit, bad cut, bomb hit) or when it passes the hit plane (miss).

## 3. WebSocket and REST (Person 2)

### REST

| Endpoint | Returns |
|---|---|
| `GET /api/games` | `[{ "id": "saber", "label": "Fly Saber" }]` |
| `GET /api/brains` | `[{ "id": "real", "label": "Real fly brain", "description": "..." }]`: always `expert` (the scripted player), then each brain mode in `configs/brains.yaml` (section 4, Brain modes) whose `bundles/<id>/` holds a trained checkpoint at startup. View-only modes are not listed: they have nothing to play |
| `GET /api/maps` | `[{ map_id, split, song: {title, artist, mapper, license}, difficulty: {name}, note_count }]`; only maps with allowlisted audio |
| `GET /api/maps/{map_id}` | the full normalized map (section 1) |
| `GET /api/maps/{map_id}/audio` | the audio file; 404 unless the map is in `demo_allowlist.json` |
| `GET /api/explorer/{brain}` | `explorer.json` (Person 1) |
| `GET /api/eye_lattice` | `{ "left": [[azimuth_deg, elevation_deg], ...], "right": [...] }`, the order used by eye images |

### WebSocket: `WS /ws/session`

Text messages are JSON objects with a `type` field. Binary messages carry activity and eye images (layout below). One socket is one session. Versus runs two brains in one session on identically seeded envs; every per-brain message has a `slot` (0 or 1).

#### Client to server

| `type` | Fields | Notes |
|---|---|---|
| `start` | `game`, `brains: string[]` (1 or 2), `map_id`, `seed: int` | Loads the session. Server answers `ready`; the sim does not step yet. |
| `play` | `song_t: float` | Sent the moment the client starts audio. Sim starts stepping from `song_t`. |
| `clock` | `song_t: float` | Sent about once a second from `AudioContext.currentTime`. Server resyncs the sim to it (spec 14). |
| `pause` | `paused: bool` | Client pauses audio at the same time. |
| `reset` | | Back to `ready` state, same map and brains. |
| `set_interventions` | `slot`, `interventions` | `interventions` schema is section 4. Applies immediately, no retraining. |
| `human_action` | `color`, `cell`, `cut` | You vs. the Fly (stretch). Same encoding as a saber in `render_state`. |
| `narrate` | `slot` | Ask for an explanation of the last miss (Person 3). |

Server behaviour:

- Each tick, the server steps every slot until its sim time reaches the client's song time (the `play` or latest `clock` value plus wall time elapsed since), then sends one `frame` per slot that stepped. A `clock` ahead of the sim makes it catch up at once; a `clock` behind it holds the sim until the audio catches up.
- `start` with an unknown game, brain or map, `play` or `reset` before `start`, and malformed messages get an `error`; the socket stays open.
- `set_interventions`, `human_action` and `narrate` answer `error` until they are implemented.
- Brains play greedily from their checkpoint, with each slot's membrane potentials carried frame to frame and reset to the rest state on `play` and `reset`. Brain noise is seeded by `start.seed`, so a replay of the same session is identical.
- Each served brain loads from its bundle (`training.play_eval.load_trained` with the bundle's `subgraph/`), never from a local subgraph build. A bundle that fails to load (missing file, fingerprint mismatch) is logged and left out of `/api/brains`; the server still starts.
- The audio key for `/api/maps/{map_id}/audio` and the allowlist is `map_id` up to its last `_`, so all difficulties of a song share one file.

#### Server to client

| `type` | Fields | Rate |
|---|---|---|
| `ready` | `session_id`, `game`, `map_id`, `slots: [{slot, brain}]`, `step_hz`, `n_neurons`, `cell_types: string[]` | once per `start` / `reset`. `n_neurons` and `cell_types` are the slots' brain subgraph (same list and order as `neurons.json`); 0 and `[]` when every slot is `expert` |
| `frame` | `slot`, `render_state` (section 2) | 30 Hz per slot |
| `event` | `slot`, `t`, `kind`, `note_id` | as they happen |
| `ended` | `slot`, `score` (same shape as `render_state.score`) | end of song |
| `narration` | `slot`, `text` | on request (Person 3) |
| `error` | `message` | |

`event.kind`: `hit`, `miss` (note passed the hit plane unhit), `bad_cut` (right cell, wrong direction or wrong color), `bomb` (saber hit a bomb). A hit is the correct-color saber in the note's cell with the correct direction within `saber.yaml` `timing_window_sec` (default 0.1) of `time_sec`.

#### Binary messages

Every binary message starts with a 4-byte header:

| Byte | Meaning |
|---|---|
| 0 | kind: 1 neuron rates, 2 cell-type means, 3 eye image |
| 1 | slot |
| 2 to 3 | reserved, 0 |

Then a little-endian `uint32` sim `step`, then the payload.

- Kinds 1 and 2: payload layout is section 6 (Person 1).
- Kind 3, eye image, 15 Hz: `uint16` ommatidia count for the left eye, `uint16` for the right, then one `uint8` luminance per ommatidium, left eye first, in the lattice order from `GET /api/eye_lattice` (served once with each ommatidium's azimuth and elevation in degrees, so the front end can draw the hex mosaic).

## 4. Subgraph files, `groups.json`, and interventions (Person 1)

Built by `uv run python scripts/build_subgraph.py` into `data/processed/malecns_subgraph/`. Neuron **index** (row number, 0 to N-1) is the id used everywhere: edges, groups, activity stream, interventions. Body ids are only for lookup.

**`neurons.parquet`**, one row per neuron, sorted by role then type then side:

| Column | Type | Meaning |
|---|---|---|
| (row index) | int | subgraph index |
| `body_id` | int64 | MaleCNS body id |
| `type` | str | cell type, e.g. `T4a`, `LPLC2`, `DNp01` |
| `superclass` | str | e.g. `ol_intrinsic`, `visual_projection`, `descending_neuron` |
| `role` | str | `columnar`, `projection`, `descending`, `intermediate`, `modulatory` (octopamine, dopamine, serotonin; the mood system) |
| `side` | str | `L`, `R`, `M` (rarely missing) |
| `x`, `y`, `z` | float64 | soma position in 8 nm voxels; NaN for 144 neurons |
| `position_inherited` | bool | true if position was copied from the strongest partner (photoreceptors mostly) |
| `hex1`, `hex2` | float64 | eye column, about 1 to 36 and 1 to 39, per side. Set for every `columnar` neuron, NaN otherwise |
| `nt` | str | consensus neurotransmitter |
| `sign` | int8 | +1, -1, or 0 |

**`edges.parquet`**: `pre` (int32 index), `post` (int32 index), `weight` (int32 synapse count, at least 5), `sign` (int8, sign of `pre`). Sorted by `post` then `pre`.

**`groups.json`**: `{"group_name": [index, ...]}`, sorted indices, empty groups omitted. Current groups: `photoreceptors`, `lamina`, `medulla`, `T4`, `T5`, `motion_detectors`, `LPLC2`, `LC4`, `looming_detectors` (LPLC2, LC4, LPLC1, LPLC4), `LC_all`, `lobula_plate_tangential`, `giant_fiber`, `DN_all`, `intermediates`, `eye_left`, `eye_right` (photoreceptors plus lamina), `octopamine`, `dopamine`, `serotonin` (modulatory-role neurons, what the mood sliders drive), `modulatory_<nt>` (every neuron with that transmitter, including a serotonergic DN), plus `_left` and `_right` versions of `photoreceptors`, `lamina`, `T4`, `T5`, `LPLC2`, `LC_all`. Groups and type lists live in `brain/connectome/groups.py`.

**`meta.json`**: counts, sign totals, and the config used.

### Front end copies: `frontend/public/data/brain/`

Committed to git so the front end needs no Python or data download. Regenerate with `make subgraph export-web` and commit. The browser loads `fetch("/data/brain/neurons.json")` and `fetch("/data/brain/groups.json")`.

**`groups.json`**: identical to the one above.

**`neurons.json`**: struct of arrays. Position `i` in every per-neuron array is neuron index `i`, the same index used by groups, interventions, and the activity stream.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `n_neurons` | int | N (19,462) |
| `cell_types` | string[] | sorted unique types. **Cell-type id = index in this list**, the same list and order as `ready.cell_types` and activity kind 2 |
| `superclasses`, `roles`, `sides`, `nts`, `position_sources` | string[] | lookup tables for the coded arrays below |
| `bounds_um` | `{min: [x,y,z], max: [x,y,z]}` | bounding box of all positions, micrometers |
| `type` | int[N] | cell-type id |
| `superclass`, `role`, `side`, `nt` | int[N] | index into the matching lookup table |
| `sign` | int[N] | +1 excitatory, -1 inhibitory, 0 modulatory or unclear |
| `body_id` | int[N] | MaleCNS body id, for linking out |
| `position_um` | float[3N] | flat `x0, y0, z0, x1, ...` in micrometers (drop straight into a `Float32Array` for instancing) |
| `position_source` | int[N] | 0 soma, 1 copied from strongest partner, 2 mean of its cell type (no position in the data) |

Coordinates are MaleCNS space: the whole CNS including the ventral nerve cord, about 730 x 470 x 930 um. Center on `bounds_um` for the camera.

**Eye input:** the brain takes `eye` as `(2, n_columns)`, one value per kept eye column (437) in the order of `columns.json` below. Row 0 is luminance and drives R1 to R6; in columns with no traced R1 to R6 (most of them, see DATA_NOTES) it drives that column's L1 and L2 with inverted sign instead. Row 1 is color (red notes high, blue low, background 0.5) and drives R7 and R8. The env should render the eye at the column directions in `columns.json`, so its `obs` is exactly this array.

**`columns.json`** (`frontend/public/data/brain/columns.json`, 10 KB): struct of arrays, position `i` is column `i` (`eye[:, i]`).

| Field | Type | Meaning |
|---|---|---|
| `n_columns` | int | 437 (217 left, 220 right) |
| `side` | string[] | `L` or `R` |
| `hex1`, `hex2` | int[] | MaleCNS column coordinates (even numbers only: every 2nd column is kept) |
| `azimuth_deg`, `elevation_deg` | float[] | viewing direction, `fly_saber/eye.py` convention: azimuth positive to the fly's right, elevation positive up |
| `luminance_input` | int[] | 0 a traced R1-R6 takes luminance, 1 the L1/L2 fallback does (index into `luminance_inputs`) |
| `spacing_deg`, `kept_spacing_deg` | float | 5 between neighboring real columns (from `configs/eye.yaml`); 10 between kept columns |

Angles: hex1 and hex2 are axial hex coordinates with axes 120 degrees apart. **+hex1 points up and forward, +hex2 up and backward**, the same in both eyes (fit to lamina soma positions, which map the eye directly). So forward = (sqrt(3)/2) * 5 * (hex1 - hex2) and up = 0.5 * 5 * (hex1 + hex2) degrees, then each eye is shifted so its most frontal column is 10 degrees past the midline (`binocular_overlap_deg`) and its vertical midpoint is the equator. Result: each eye spans about -129 to +10 degrees azimuth (mirrored for the right eye) and -80 to +80 degrees elevation. The optic chiasm flips front and back between lamina and medulla, but the hex coordinates name columns, and the connectome wiring already carries the flip, so **the renderer must not flip anything**. The absolute angles are an idealized lattice; the orientation is measured. Kept columns are 10 degrees apart, so the acceptance angle should be about 10 degrees, or the fly sees gaps.


**Interventions** (`brain/model/interventions.py`): a `NamedTuple` pytree passed into every step, so the server can change it live without recompiling.

| Field | Shape | Meaning |
|---|---|---|
| `gain` | (N,) | output multiplier; 0 lesions the neuron (its rate and its synaptic output) |
| `opto` | (N,) | extra input current; around 5 forces a neuron on |
| `noise_std` | () | std of current noise on every neuron each substep (for the arousal slider) |

Helpers: `no_interventions(N)`, `lesion(iv, groups["T4"])`, `stimulate(iv, groups["giant_fiber"], 5.0)`.

**`set_interventions` payload** (the `interventions` field of the WebSocket message). The server builds the pytree with `brain.model.controls.from_controls(payload, groups, load_controls(), n_neurons)`, which validates names and clamps values (ValueError on bad input, answer `error`). Every key is optional; `{}` is the unmodified brain. Each message replaces the previous state, it does not add to it.

```jsonc
{
  "lesion": ["looming_detectors", "eye_left"],   // group names: silence them
  "stimulate": { "giant_fiber": 5.0 },          // group -> opto current, 0 to stimulation.max
  "mood": { "arousal": 0.8, "calm": -0.3 },     // mood name -> level, -1 to 1, 0 neutral
  "noise": 0.1                                  // neural noise std, 0 to noise.max
}
```

Lesion wins over stimulation and mood for neurons in both. Any group in `groups.json` is accepted; the curated ones are in `controls.json`.

**`controls.json`** (`frontend/public/data/brain/controls.json`, from `configs/controls.yaml`): what the UI shows.

- `sections`: `[{group, label, description, n_neurons}]`, 12 lesionable brain sections (each eye, motion detectors, looming detectors, LC feature detectors, HS/VS, central relay, giant fiber, all DNs, octopamine, dopamine, serotonin).
- `moods`: `[{name, label, group, low_label, high_label, description, min: -1, max: 1, default: 0}]`: `arousal` (octopamine), `reward_drive` (dopamine), `calm` (serotonin).
- `stimulation: {min, max}`, `noise: {min, max, default}`.

**How moods work:** a mood level drives its modulatory group like opto. Each neuron those modulatory neurons synapse onto has its output multiplied by exp(gain * change in their rate from rest), with `modulation_gain` per transmitter in `configs/model.yaml` (octopamine +1, dopamine +1, serotonin -1, an assumption) and capped at e. At level 0 the factor is exactly 1, so moods never need retraining. -1 silences the group (same as lesioning it). Targets are mostly central relay neurons, visual projection neurons (LC9, LC14), and DNs; in this data the modulatory neurons do not synapse directly onto T4/T5.

**Effect on motor output before training** (`uv run python scripts/check_controls.py`, mean change in DN rates, looming stimulus): arousal +1 97%, reward drive +1 49%, noise 0.5 36%, stimulate looming detectors 22%, lesion central relay 19%, lesion octopamine 16%, lesion dopamine 10%, calm +-1 about 6%, lesion motion detectors 4%, lesion looming detectors 4%, lesion one eye under 1%. Vision moves DNs little until training strengthens it; rerun with a checkpoint (`check_controls.py checkpoints/.../bc_N.npz`) after training.

### Brain modes (`scripts/export_brains.py`, `configs/brains.yaml`)

Four brains the demo can switch between, all with the **same neurons** (one `neurons.json`), different wiring:

| `id` | What | Playable |
|---|---|---|
| `real` | The fly's own wiring | yes |
| `shuffled_layers` | Same layer-to-layer connection counts, random partners within each layer pair | yes |
| `shuffled_signs` | Real wiring, excitation and inhibition scrambled | yes |
| `shuffled_free` | Partners from anywhere: 1,068 direct eye-to-motor connections versus 1 | view only (no checkpoint) |

Run on HiPerGator (it needs the checkpoints and their subgraphs). It writes:

- **`bundles/<id>/`** for the server: `bc_<n>.npz` plus `subgraph/` (exact training files), `MANIFEST.json` (`checkpoint` is null for view-only). The server's `GET /api/brains` should list these, and load each from its bundle.
- **`frontend/public/data/brains/index.json`**: `{schema_version, modes: [{id, label, description, playable, bundle, play, wiring}]}`. `play` is `{heldout, train}` hit rates (from `eval_<ckpt>.json`) or null. `wiring` has `direct_eye_to_motor_connections`, `eye_to_motor_synapses_median`, `eye_to_motor_synapses_histogram` (synapses: motor neuron count), `motor_neurons_unreachable_from_eye`, `connections_shared_with_real` (0 to 1), `excitatory_neuron_fraction`.
- **`frontend/public/data/brains/<id>.json`**: `{id, sign: int[N], edges_columns: ["pre","post","weight","sign"], edges: [[pre, post, weight, sign], ...]}`. `sign` replaces `neurons.json`'s `sign` for that mode (only `shuffled_signs` differs). `edges` is the strongest 60 connections of each layer-to-layer block (about 2,500), neuron indices into `neurons.json`, for drawing the wiring in 3D: in `shuffled_free` many lines run straight from the eyes to the motor neurons.

## 5. Brain step and checkpoints (Person 1)

`brain/model/brain.py`, configured by `configs/model.yaml`:

```python
graph, columns, n = load_graph(cfg)  # columns: eye column table, one row per column
params = init_params(key, graph, cfg, n)
step = make_step(graph, cfg, n)  # pure function: jit it, vmap it over envs
v = init_state(n)  # reset at episode start
v, out = step(params, v, eye, interventions, key)
logits_per_head = split_heads(out.logits, cfg.action_heads)
```

| Argument / output | Shape | Meaning |
|---|---|---|
| `eye` | (2, n_columns) float in [0, 1] | row 0 luminance, row 1 color, per eye column in `columns.json` order (437 columns) |
| `v` | (N,) | membrane potentials; carry between frames, reset per episode |
| `key` | PRNGKey | for noise; pass a fresh one per frame |
| `out.logits` | (sum of `action_heads`,) | `[13, 5, 13, 5]`: red position (12 cells + idle), red direction (4 + no_cut), blue position, blue direction |
| `out.value` | () | PPO value estimate |
| `out.rates` | (N,) | all neuron rates, for the activity stream |

One call is one 30 Hz game frame (4 Euler substeps of 8.3 ms). To vmap over envs: `jax.vmap(step, in_axes=(None, 0, 0, None, 0))`.

**Readout and training (status).** The first HiPerGator BC run (lr 1e-3, raw DN readout) sat exactly at the no-vision base rate (loss 4.4, position accuracy 37%, hit rate 0) for 300 updates. Diagnosis:

- Visual information does reach the DNs: a least-squares readout of the *untrained* brain's DN rates predicts the expert's next cell on held-out songs at 59% (base rate 27%; eye pixels 79%, projection neurons 68%). Gradient descent finds that readout very slowly because DN rates are highly correlated.
- So `training/calibration.py` **fits the readout** before BC: it plays 64 synthetic songs with the expert, sets `dn_mean`/`dn_std` (fixed params, saved in checkpoints), and solves ridge regression per head plus a temperature. The step reads `clip((rate - dn_mean) / dn_std, +-5) / sqrt(D)`.
- Episodes start from `rest_state` (brain settled on the gray background), not zeros; `BrainPolicy.rest` holds it. After loading a checkpoint, recompute it.
- Result with no gradient training: greedy play hits **18% of notes on training songs and 16% on the held-out song**, versus 0% before.
- The readout reads DN ripples of about 1e-5. With a fixed baseline, updating the brain's own parameters washed them out (brain lr 1e-5: loss 3.4 to 97, position accuracy 57% to 24% in ten updates).
- **Live baseline** (`live_baseline: true`): every update, `BrainPolicy.prepare` sets `dn_mean` to the DN rates of the brain settled on the gray background under the *current* params (`rest_dn`, 45 frames, one env, with gradients). Drift in the brain's operating point now cancels in the readout. Interventions are not applied to the baseline, so lesions and moods still reach the actions. The readout fit uses the same baseline (centered ridge, intercept folded into the bias). Optimizers keep the stored `dn_mean`/`dn_std` fixed.
- With it the brain trains without collapsing: brain lr 1e-5 kept position accuracy at 57% to 62% over 40 small local updates (1e-4 also stable, noisier loss). Defaults: `freeze_brain: false`, `brain_learning_rate: 1e-5`. Whether brain training beats readout-only fine-tuning (about 62% at the same point) needs a longer GPU run.

**Training memory:** `remat: true` (default) wraps the step in `jax.checkpoint`, so backprop through a rollout stores each frame's inputs and recomputes the substeps. Measured for one BC update (64 envs, 10 s windows): 169 GB without it, 2.5 GB with it; memory grows linearly with envs times window length. Gradients are identical (tested). Activation is `relu_sat` (relu that levels off at `rate_max` 10), so excitatory loops cannot run away when moods raise gains. Untrained the network is stable and responsive: about 97% of neurons tonically active, looming in the eye changes LPLC2 and LC4 by about 2%, and DNs by about 0.05% before training. Speed on a laptop CPU: 4 ms per frame at batch 1, about 1,800 frames/s at batch 64. Run `uv run python scripts/bench_brain.py` to recheck.

**Checkpoints** (`training/checkpoints.py`): one `.npz` per checkpoint, at `checkpoints/{variant}/seed{seed}/{stage}_{step}.npz` (gitignored), for example `checkpoints/real/seed0/bc_2000.npz`.

```python
from training.checkpoints import save_checkpoint, load_checkpoint, subgraph_fingerprint

fp = subgraph_fingerprint(model_cfg.subgraph_dir)
save_checkpoint(
    path,
    params,
    variant="real",
    stage="bc",
    step=i,
    configs={"model": model_cfg, "train": bc_cfg},
    subgraph=fp,
    metrics=row,
)
params, meta = load_checkpoint(path, init_params(key, graph, model_cfg, n), expect_subgraph=fp)
```

- Arrays are keyed by pytree path (`.log_tau`, `.readout_w`, ...); a `__meta__` entry holds JSON: `format_version`, `variant`, `stage` (`bc` or `ppo`), `step`, `config_hash` (12 hex chars over all `configs`), `configs`, `subgraph` (neuron, edge, synapse counts), `metrics`.
- Loading takes a template pytree (any policy: brain `Params`, MLP, stub) and raises if keys or shapes differ, or if the checkpoint was trained on a different subgraph. Writes are atomic, so the server never reads a half-written file.
- About 0.7 MB for the brain.

### Running the trained fly in the server (Person 2)

Everything a live session needs, in order. The bundle is `bundles/<id>/` (demo fly: `bundles/real_fast/`, or a brain mode from `scripts/export_brains.py`).

```python
from brain.model.controls import from_controls, load_controls
from training.play_eval import load_trained

fly = load_trained("bundles/real_fast/bc_950.npz", subgraph_dir="bundles/real_fast/subgraph")
# fly.env (SaberEnv seeing through the brain's eye columns), fly.policy (BrainPolicy),
# fly.params (trained, baseline applied), fly.groups, fly.n_neurons
step = jax.jit(fly.policy.step)          # one 30 Hz frame; warm up once at startup (~11 s compile)
v = fly.policy.init_carry()              # start of every song: the brain at rest, not zeros
params = fly.params

# On every set_interventions message:
iv = from_controls(msg.interventions, fly.groups, load_controls(), fly.n_neurons)  # ValueError -> error
params = fly.policy.prepare(fly.params, iv)   # re-center the readout on the modified brain
policy_iv = iv

# Every frame:
v, out = step(params, v, obs, policy_iv, key)   # obs = env observation, (2, 437)
action = greedy over split_heads(out.logits, [13, 5, 13, 5]) -> env.step
# out.rates (N,) feeds the activity stream (section 6), neuron i = neurons.json index i
```

- **Always call `prepare(params, iv)` when interventions change.** Without it the readout keeps the untouched brain's baseline, and every control (even lesioning the 2 giant fibers) collapses play to 0 to 4% hits. With it, a lesion changes how vision drives the motor neurons. Cost: about 45 brain frames, once per change.
- **Use the bundle's `subgraph/`, never a local rebuild.** HiPerGator's subgraph has 19,463 neurons; a local build had 19,462. `load_trained` refuses a mismatched subgraph.
- Front-end data that matches: `frontend/public/data/brain/` (re-exported from HiPerGator in #30).
- Versus: load two bundles (two `load_trained` calls), same `env`, same episode seed.

## 6. Activity stream (Person 1)

Spec 13: uint8-quantized rates for all subgraph neurons at 15 Hz, per-cell-type means at 30 Hz. Pin the binary layout here.
