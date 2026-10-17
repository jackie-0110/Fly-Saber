# Contracts and WebSocket Protocol

Fill these in during the hour 0 contracts session (30 minutes, all three). Everyone builds against these with fake data until the real pieces land. Change a contract only after telling its consumers.

| Contract | Owner | Consumers | Status |
|---|---|---|---|
| Normalized map schema | Person 2 | 1, 3 | TODO |
| `render_state` format | Person 2 | 3 | TODO |
| WebSocket protocol (both directions) | Person 2 | 3 | TODO |
| `groups.json` and interventions schema | Person 1 | 2, 3 | groups done, interventions TODO |
| Brain step function signature and checkpoint format | Person 1 | 2, 3 | TODO |
| Activity stream format (quantized rates, cell-type means) | Person 1 | 3 | TODO |

## 1. Normalized map schema (Person 2)

Starting point from spec 8.2: notes `{time_sec, lane (0-3), layer (0-2), color, cut_direction, is_bomb}`, walls `{time_sec, duration_sec, lane, width}`, plus song metadata. Pin exact types, enums, and file layout here.

## 2. `render_state` (Person 2)

## 3. WebSocket messages (Person 2)

Client to server (spec 13): `start`, `set_interventions`, `human_action`, `pause`, `reset`, `narrate`.

Server to client: `frame`, `activity` (binary), `eye`, `event`, `narration`.

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
