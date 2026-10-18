"""Named neuron groups for interventions and the UI (spec 5.4).

Owner: Person 1. Output format is the groups.json contract in docs/reference/PROTOCOL.md.
"""

import pandas as pd

from brain.connectome.signs import MODULATORY
from brain.connectome.subgraph import match_types

# Group name -> regex list on `type`. Side-specific groups are added automatically.
TYPE_GROUPS: dict[str, list[str]] = {
    "photoreceptors": ["R1-R6", "R7.*", "R8.*"],
    "lamina": ["L[1-5]"],
    "medulla": ["Mi\\d+", "Tm\\d+.*"],
    "T4": ["T4[a-d]"],
    "T5": ["T5[a-d]"],
    "motion_detectors": ["T4[a-d]", "T5[a-d]"],
    "LPLC2": ["LPLC2"],
    "LC4": ["LC4"],
    "looming_detectors": ["LPLC2", "LC4", "LPLC1", "LPLC4"],
    "LC_all": ["LC\\d+.*"],
    "lobula_plate_tangential": ["HS.*", "VS.*"],
    "giant_fiber": ["DNp01"],
}
SIDED = ("photoreceptors", "lamina", "T4", "T5", "LPLC2", "LC_all")


def build_groups(neurons: pd.DataFrame) -> dict[str, list[int]]:
    """Group name -> sorted subgraph indices. Empty groups are dropped."""
    groups: dict[str, pd.Index] = {}
    for name, patterns in TYPE_GROUPS.items():
        groups[name] = neurons.index[match_types(neurons["type"], patterns)]
    for name in SIDED:
        for side, label in (("L", "left"), ("R", "right")):
            groups[f"{name}_{label}"] = groups[name].intersection(
                neurons.index[neurons["side"] == side]
            )
    groups["eye_left"] = groups["photoreceptors_left"].union(groups["lamina_left"])
    groups["eye_right"] = groups["photoreceptors_right"].union(groups["lamina_right"])
    groups["DN_all"] = neurons.index[neurons["role"] == "descending"]
    for nt in MODULATORY:
        groups[nt] = neurons.index[(neurons["role"] == "modulatory") & (neurons["nt"] == nt)]
    groups["intermediates"] = neurons.index[neurons["role"] == "intermediate"]
    for nt in MODULATORY:
        groups[f"modulatory_{nt}"] = neurons.index[neurons["nt"] == nt]
    return {k: sorted(int(i) for i in v) for k, v in groups.items() if len(v)}
