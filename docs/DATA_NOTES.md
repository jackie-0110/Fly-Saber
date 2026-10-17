# Data Notes

What the real files actually contain, and every deviation from the spec. Inspect before coding against anything (spec ground rule 2).

## MaleCNS v1.0 (Person 1)

### Source

Public bulk files, no token needed (neuPrint token only needed for API queries):

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>
```

| File | Size | Used for |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14 MB | neuron identity, type, side, soma position, hex column |
| `body-neurotransmitters-male-cns-v1.0.feather` | 41 MB | neurotransmitter per body |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | edges (pre, post, synapse count) |

Saved under `data/raw/malecns/` (gitignored). Listing: https://male-cns.janelia.org/download/

### Annotations: 211,577 rows, 36 columns

Columns we use:

| Spec field | Real column | Notes |
|---|---|---|
| id | `bodyId` (int64) | |
| cell type | `type` (str) | 164,506 rows have a type |
| coarser class | `superclass` (str) | e.g. `ol_intrinsic`, `visual_projection`, `ol_sensory`, `descending_neuron` |
| side | `somaSide` (`L`/`R`/`M`), fallback `rootSide` | photoreceptors have no `somaSide`; use `rootSide` |
| 3D position | `somaLocation` ([x, y, z] list, 8 nm voxels) | missing for most photoreceptors (13 of 3,377 R1-R6), see below |
| retinotopic column | `assignedOlHex1`, `assignedOlHex2` (float, 1 to 36 and 1 to 39) | 892 unique columns; set for lamina and medulla columnar types (L1-L5, Mi1, Mi4, Mi9, Tm1, Tm2, Tm9, Tm20, T1, C3), **not** for photoreceptors, T4, T5 |

Also present and possibly useful: `instance` (e.g. `DNp01(GF)_R`), `flywireType`, `hemibrainType`, `subclass`, `status` (`Traced` for 165,122 rows; filter to these).

### Neurotransmitters: 1,835,518 rows

Keyed by `body`. Use `consensus_nt` (falls back to cell-type level). Values: `acetylcholine` 104k, `glutamate` 29k, `gaba` 22k, `histamine` 8k, `dopamine` 396, `octopamine` 101, `serotonin` 48, `unclear` for the rest (mostly untyped fragments). 164,446 of 164,506 typed bodies have a row.

### Key cell type counts (typed neurons, both sides)

| Group | Type names in data | Neurons |
|---|---|---|
| Photoreceptors R1-R6 | `R1-R6` | 3,377 |
| R7 | `R7p`, `R7y`, `R7d`, `R7_unclear`, `R7R8_unclear` | 1,385 |
| R8 | `R8p`, `R8y`, `R8d`, `R8_unclear` | 1,329 |
| Lamina | `L1` to `L5` | 8,884 |
| Mi | 13 types | 9,589 |
| Tm | 29 types | 19,861 |
| T4 | `T4a` to `T4d` | 6,861 |
| T5 | `T5a` to `T5d` | 6,719 |
| LPLC2 | `LPLC2` | 185 |
| LC | 48 types | 4,253 |
| HS, VS | 4 types each | 8, 34 |
| Descending | 484 `DN*` types | 1,342 |
| Giant fiber | `DNp01` (instance `DNp01(GF)_L/R`) | 2 |

### Deviations from the spec

1. **Histamine must be inhibitory.** Photoreceptors (R1-R8) are 100% `histamine`. Spec 5.3 maps unlisted transmitters to 0, which would disconnect the eyes from the brain. Map `histamine` to `-1` (HisCl1 chloride channels; photoreceptors hyperpolarize L1 and L2). Lamina L1 is `glutamate`, so spec's `-1` applies.
2. **Photoreceptor to ommatidium mapping.** Photoreceptors have no hex column and almost no soma position. Plan: give each photoreceptor the hex column of the lamina neuron (L1/L2/L3) it sends the most synapses to. That gives a real retinotopic map instead of the nearest-angle projection fallback in spec 7.1. Person 2 consumes this.
3. **Positions for the 3D view.** Photoreceptors and some others lack `somaLocation`. Fallback: position of their main lamina partner, or synapse centroid (would need `body-stats` or synapse files). TBD.
4. **Neuron budget: every 2nd column.** Taking every column of the core columnar types is about 50k neurons, over the 5k to 20k budget. We keep columns where both hex coordinates are even (`column_stride: 2`), about 220 of 892 per eye. Decided by the team.
5. **T4 and T5 have no hex column** in the data either. They inherit one the same way as photoreceptors (strongest hex-assigned partner, 2 passes, optic lobe superclasses only). Every kept columnar neuron ends up with a column.
6. **Incomplete photoreceptor tracing.** Only 3,377 R1-R6 are traced (a full set would be about 6 per column times 892 columns per eye), and 84% of those with a lamina partner get a column. In the subgraph, 167 (left) and 183 (right) of about 220 sampled columns per eye have a photoreceptor. Some lamina neurons (L3, L4, L2) have no inputs at all in the subgraph. Workaround: in columns without a photoreceptor, the eye drives L1 and L2 directly with inverted sign.

### Subgraph result (`configs/subgraph.yaml`, `min_synapses: 5`)

| | Count |
|---|---|
| Neurons | 19,403 (10,122 columnar, 5,471 projection, 2,500 intermediate, 1,310 descending) |
| Edges | 336,983 carrying 6.1M synapses |
| Edge signs | 254k excitatory, 76k inhibitory, 7k zero (octopamine, dopamine, serotonin, unclear) |
| Positions | 2,793 inherited from a partner, 144 missing |
| DNs reachable from photoreceptors | 1,288 of 1,310, mostly 4 to 5 synapses (photoreceptor, lamina, medulla, T4/T5 or LC, central bridge, DN) |

Each kept neuron gets only part of its real input from inside the subgraph (median share: columnar 0.30, projection 0.20, intermediate 0.44, descending 0.54). The rest comes from neurons we cut. Trainable per-neuron biases stand in for that missing drive.

Intermediates are the top 2,500 central neurons by min(synapses in from projection neurons, synapses out to DNs); a neuron needs both.

## FlyWire (Person 1)

Not downloaded yet.

## Ommatidia to photoreceptor mapping (Person 2)

See MaleCNS deviation 2. Hex grid is about 36 by 39 per eye, 892 columns.

## Beat Saber map formats, v2 and v3 (Person 2)
