# Contracts and WebSocket Protocol

Fill these in during the hour 0 contracts session (30 minutes, all three). Everyone builds against these with fake data until the real pieces land. Change a contract only after telling its consumers.

| Contract | Owner | Consumers | Status |
|---|---|---|---|
| Normalized map schema | Person 2 | 1, 3 | Draft |
| `render_state` format | Person 2 | 3 | Draft |
| WebSocket protocol (both directions) | Person 2 | 3 | Draft |
| `groups.json` and interventions schema | Person 1 | 2, 3 | groups done, interventions TODO |
| Brain step function signature and checkpoint format | Person 1 | 2, 3 | TODO |
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
- `obs`: float32 `(2, n_ommatidia)`, values 0 to 1, lattice order of `fly_saber.eye.lattice` (left eye first). Channel 0 is luminance, for R1 to R6. Channel 1 is color (red notes high, blue low, background 0.5), for R7 and R8.
- `action`: int32 `(n_sabers, 2)`, rows red then blue: `[cell, cut]`, cell 0 to 11 or 12 (idle), cut 0 up, 1 down, 2 left, 3 right, or 4 (no cut). That is a 13-way position head and a 5-way direction head per saber. `n_sabers` is 1 when `saber.yaml` sets `two_sabers: false`.
- `fly_saber.experts.saber_expert(env, state)` returns the perfect action, for behavior cloning.

A note leaves the scene when an `event` for its `id` arrives (hit, bad cut, bomb hit) or when it passes the hit plane (miss).

## 3. WebSocket and REST (Person 2)

### REST

| Endpoint | Returns |
|---|---|
| `GET /api/games` | `[{ "id": "saber", "label": "Fly Saber" }]` |
| `GET /api/brains` | `[{ "id": "real", "label": "Real wiring", "description": "..." }]`, one per variant in `variants.yaml` that has a checkpoint |
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
- The audio key for `/api/maps/{map_id}/audio` and the allowlist is `map_id` up to its last `_`, so all difficulties of a song share one file.

#### Server to client

| `type` | Fields | Rate |
|---|---|---|
| `ready` | `session_id`, `game`, `map_id`, `slots: [{slot, brain}]`, `step_hz`, `n_neurons`, `cell_types: string[]` | once per `start` / `reset` |
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
| `role` | str | `columnar`, `projection`, `descending`, `intermediate` |
| `side` | str | `L`, `R`, `M` (rarely missing) |
| `x`, `y`, `z` | float64 | soma position in 8 nm voxels; NaN for 144 neurons |
| `position_inherited` | bool | true if position was copied from the strongest partner (photoreceptors mostly) |
| `hex1`, `hex2` | float64 | eye column, about 1 to 36 and 1 to 39, per side. Set for every `columnar` neuron, NaN otherwise |
| `nt` | str | consensus neurotransmitter |
| `sign` | int8 | +1, -1, or 0 |

**`edges.parquet`**: `pre` (int32 index), `post` (int32 index), `weight` (int32 synapse count, at least 5), `sign` (int8, sign of `pre`). Sorted by `post` then `pre`.

**`groups.json`**: `{"group_name": [index, ...]}`, sorted indices, empty groups omitted. Current groups: `photoreceptors`, `lamina`, `medulla`, `T4`, `T5`, `motion_detectors`, `LPLC2`, `LC4`, `looming_detectors` (LPLC2, LC4, LPLC1, LPLC4), `LC_all`, `lobula_plate_tangential`, `giant_fiber`, `DN_all`, `intermediates`, `modulatory_<nt>`, plus `_left` and `_right` versions of `photoreceptors`, `lamina`, `T4`, `T5`, `LPLC2`, `LC_all`. Groups and type lists live in `brain/connectome/groups.py`.

**`meta.json`**: counts, sign totals, and the config used.

**Eye input (to agree with Person 2):** the eye drives photoreceptors by `(side, hex1, hex2)`. About 170 to 180 of the 220 sampled columns per eye have a photoreceptor; for the rest, inject into that column's L1 and L2 with inverted sign (see DATA_NOTES).

Interventions schema: TODO (with `brain/model/interventions.py`).

## 5. Brain step and checkpoints (Person 1)

## 6. Activity stream (Person 1)

Spec 13: uint8-quantized rates for all subgraph neurons at 15 Hz, per-cell-type means at 30 Hz. Pin the binary layout here.
