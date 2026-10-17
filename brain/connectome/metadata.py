"""Brain data for the front end and the Brain Explorer (spec 5.5).

Owner: Person 1. Format is documented in docs/PROTOCOL.md section 4.
explorer.json (descriptions, top connections) builds on web_neurons later.
"""

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
VOXEL_UM = 0.008  # MaleCNS coordinates are 8 nm voxels
ROLES = ["columnar", "projection", "intermediate", "descending"]
SIDES = ["L", "R", "M"]
POSITION_SOURCES = ["soma", "partner", "type_mean"]


def cell_types(neurons: pd.DataFrame) -> list[str]:
    """Sorted unique cell types. Index into this list is the cell-type id everywhere."""
    return sorted(neurons["type"].unique().tolist())


def web_neurons(neurons: pd.DataFrame) -> dict[str, Any]:
    """Struct-of-arrays neuron table for the browser; array position i is neuron index i."""
    types = cell_types(neurons)
    superclasses = sorted(neurons["superclass"].unique().tolist())
    nts = sorted(neurons["nt"].unique().tolist())

    xyz = neurons[["x", "y", "z"]].astype(float)
    source = np.where(neurons["position_inherited"], 1, 0)
    missing = xyz["x"].isna().to_numpy()
    type_mean = xyz.groupby(neurons["type"]).transform("mean")
    xyz = xyz.fillna(type_mean).fillna(xyz.mean())  # types with no position at all
    source[missing] = 2
    um = (xyz.to_numpy() * VOXEL_UM).round(1)

    def codes(col: str, vocab: list[str]) -> list[int]:
        index = {v: i for i, v in enumerate(vocab)}
        return [index[v] for v in neurons[col]]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "MaleCNS v1.0 visual-to-motor subgraph",
        "n_neurons": len(neurons),
        "cell_types": types,
        "superclasses": superclasses,
        "roles": ROLES,
        "sides": SIDES,
        "nts": nts,
        "position_sources": POSITION_SOURCES,
        "bounds_um": {"min": um.min(axis=0).tolist(), "max": um.max(axis=0).tolist()},
        "body_id": neurons["body_id"].astype(int).tolist(),
        "type": codes("type", types),
        "superclass": codes("superclass", superclasses),
        "role": codes("role", ROLES),
        "side": neurons["side"].fillna("M").map({s: i for i, s in enumerate(SIDES)}).tolist(),
        "nt": codes("nt", nts),
        "sign": neurons["sign"].astype(int).tolist(),
        "position_um": um.ravel().tolist(),
        "position_source": source.tolist(),
    }
