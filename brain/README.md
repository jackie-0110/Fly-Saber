# brain/

**Owner: Person 1 (Brain and RL).** Spec sections 5 and 6.

| Path | What |
|---|---|
| `connectome/` | Load MaleCNS and FlyWire, extract the visual-to-motor subgraph, signs, variants, Brain Explorer metadata |
| `model/brain.py` | Leaky rate network over the connectome, DN readout |
| `model/mlp.py` | Matched-parameter MLP baseline |
| `model/interventions.py` | Lesion, opto, mood, eye masks as one pytree |

Produces `data/processed/{dataset}_subgraph/`, `groups.json`, `explorer.json`. Configs: `configs/subgraph.yaml`, `model.yaml`, `variants.yaml`. Scripts: `scripts/fetch_data.py`, `build_subgraph.py`, `build_variants.py`.
