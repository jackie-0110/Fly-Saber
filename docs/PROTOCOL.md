# Contracts and WebSocket Protocol

Fill these in during the hour 0 contracts session (30 minutes, all three). Everyone builds against these with fake data until the real pieces land. Change a contract only after telling its consumers.

| Contract | Owner | Consumers | Status |
|---|---|---|---|
| Normalized map schema | Person 2 | 1, 3 | TODO |
| `render_state` format | Person 2 | 3 | TODO |
| WebSocket protocol (both directions) | Person 2 | 3 | TODO |
| `groups.json` and interventions schema | Person 1 | 2, 3 | TODO |
| Brain step function signature and checkpoint format | Person 1 | 2, 3 | TODO |
| Activity stream format (quantized rates, cell-type means) | Person 1 | 3 | TODO |

## 1. Normalized map schema (Person 2)

Starting point from spec 8.2: notes `{time_sec, lane (0-3), layer (0-2), color, cut_direction, is_bomb}`, walls `{time_sec, duration_sec, lane, width}`, plus song metadata. Pin exact types, enums, and file layout here.

## 2. `render_state` (Person 2)

## 3. WebSocket messages (Person 2)

Client to server (spec 13): `start`, `set_interventions`, `human_action`, `pause`, `reset`, `narrate`.

Server to client: `frame`, `activity` (binary), `eye`, `event`, `narration`.

## 4. `groups.json` and interventions (Person 1)

## 5. Brain step and checkpoints (Person 1)

## 6. Activity stream (Person 1)

Spec 13: uint8-quantized rates for all subgraph neurons at 15 Hz, per-cell-type means at 30 Hz. Pin the binary layout here.
