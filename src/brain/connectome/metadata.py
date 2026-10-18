"""Brain data for the front end, the fly eye, and the Brain Explorer (spec 5.5).

Owner: Person 1. Format is documented in docs/reference/PROTOCOL.md section 4.
explorer.json (descriptions, top connections) builds on web_neurons later.
"""

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
VOXEL_UM = 0.008  # MaleCNS coordinates are 8 nm voxels
ROLES = ["columnar", "projection", "intermediate", "descending", "modulatory"]
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


def web_edges(neurons: pd.DataFrame, edges: pd.DataFrame, source: str) -> dict[str, Any]:
    """Complete directed anatomy, in parquet row order; weights are synapse counts.

    Body IDs bind the endpoints to the exact neuron ordering, including when two
    subgraphs happen to have the same number of neurons. These are anatomical
    connections, not the normalized or learned efficacies used by the rate model.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "n_neurons": len(neurons),
        "n_edges": len(edges),
        "n_synapses": int(edges["weight"].sum()),
        "body_id": neurons["body_id"].astype(int).tolist(),
        **{field: edges[field].astype(int).tolist() for field in ("pre", "post", "weight", "sign")},
    }


def column_angles(
    columns: pd.DataFrame, spacing_deg: float, binocular_overlap_deg: float
) -> pd.DataFrame:
    """Viewing direction of each eye column, in the eye convention of src/fly_saber/eye.py.

    Azimuth is positive to the fly's right, elevation positive up, both in degrees.
    hex1 and hex2 are axial hex coordinates with axes 120 degrees apart: +hex1
    points up and forward, +hex2 up and backward (fit to lamina soma positions,
    see docs/reference/DATA_NOTES.md). Each eye's most frontal column sits at
    `binocular_overlap_deg` past the midline, and its vertical midpoint is the
    equator. Absolute angles are an idealized lattice, not measured optics.
    """
    h1, h2 = columns["hex1"].to_numpy(float), columns["hex2"].to_numpy(float)
    forward = np.sqrt(3) / 2 * spacing_deg * (h1 - h2)
    up = 0.5 * spacing_deg * (h1 + h2)
    azimuth, elevation = np.empty(len(columns)), np.empty(len(columns))
    for side, mirror in (("L", 1.0), ("R", -1.0)):
        m = (columns["side"] == side).to_numpy()
        if not m.any():
            continue
        behind = forward[m].max() - forward[m]
        azimuth[m] = mirror * (binocular_overlap_deg - behind)
        elevation[m] = up[m] - (up[m].max() + up[m].min()) / 2
    return pd.DataFrame({"azimuth_deg": azimuth, "elevation_deg": elevation}, index=columns.index)


def web_columns(
    columns: pd.DataFrame,
    inputs: pd.DataFrame,
    spacing_deg: float,
    binocular_overlap_deg: float,
) -> dict[str, Any]:
    """Eye column table: position i is eye input column i (the brain's eye[:, i])."""
    angles = column_angles(columns, spacing_deg, binocular_overlap_deg).round(2)
    lit = set(inputs.loc[(inputs["channel"] == 0) & (inputs["sign"] > 0), "column"])
    return {
        "schema_version": SCHEMA_VERSION,
        "n_columns": len(columns),
        "order": "sorted by side, hex1, hex2; index i is eye[:, i] in the brain step",
        "spacing_deg": spacing_deg,
        "kept_spacing_deg": 2 * spacing_deg,
        "luminance_inputs": ["photoreceptor", "lamina_fallback"],
        "side": columns["side"].tolist(),
        "hex1": columns["hex1"].astype(int).tolist(),
        "hex2": columns["hex2"].astype(int).tolist(),
        "azimuth_deg": angles["azimuth_deg"].tolist(),
        "elevation_deg": angles["elevation_deg"].tolist(),
        "luminance_input": [0 if i in lit else 1 for i in range(len(columns))],
    }


def web_controls(controls: Any, groups: dict[str, list[int]]) -> dict[str, Any]:
    """Lesionable sections, mood sliders and mood presets for the UI (a
    brain.model.controls.ControlsConfig)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sections": [
            {
                "group": s.group,
                "label": s.label,
                "description": s.description,
                "n_neurons": len(groups[s.group]),
            }
            for s in controls.sections
        ],
        "moods": [
            {
                "name": m.name,
                "label": m.label,
                "group": m.group,
                "low_label": m.low_label,
                "high_label": m.high_label,
                "description": m.description,
                "min": m.range[0],
                "max": m.range[1],
                "default": 0.0,
            }
            for m in controls.moods
        ],
        "presets": [
            {"name": p.name, "label": p.label, "description": p.description, "mood": p.mood}
            for p in controls.presets
        ],
        "stimulation": {"min": 0.0, "max": controls.max_stimulation},
        "noise": {"min": 0.0, "max": controls.max_noise, "default": 0.0},
    }


def web_explorer(types: list[str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Brain Explorer descriptions (configs/cell_types.yaml) for the subgraph's cell types.

    Raises ValueError for a type the subgraph does not have (a typo would otherwise leave
    a cell type silently undescribed) or one described twice.
    """
    known = set(types)
    seen: set[str] = set()
    out = []
    for entry in entries:
        names = list(entry["types"])
        missing = [t for t in names if t not in known]
        if missing:
            raise ValueError(f"{entry['name']}: cell types {missing} are not in the subgraph")
        twice = seen.intersection(names)
        if twice:
            raise ValueError(f"{entry['name']}: cell types {sorted(twice)} described twice")
        seen.update(names)
        out.append(
            {
                "name": entry["name"],
                "types": names,
                "description": " ".join(entry["description"].split()),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "descriptions": out}


def wiring_stats(
    neurons: pd.DataFrame, edges: pd.DataFrame, real_edges: pd.DataFrame
) -> dict[str, Any]:
    """How a brain's wiring routes the eye to the motor neurons, versus the real brain."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg

    from brain.connectome.variants import neuron_layers

    layer = neuron_layers(neurons).to_numpy()
    eye = np.flatnonzero(np.isin(layer, ["photoreceptor"]) | neurons["type"].isin(["L1", "L2"]))
    dn = np.flatnonzero(neurons["role"].to_numpy() == "descending")
    n = len(neurons)
    graph = sp.csr_matrix(
        (np.ones(len(edges)), (edges["pre"].to_numpy(), edges["post"].to_numpy())), shape=(n, n)
    )
    hops = cg.dijkstra(graph, unweighted=True, indices=eye, min_only=True)[dn]
    reached = hops[np.isfinite(hops)]
    real_pairs = set(zip(real_edges["pre"], real_edges["post"], strict=True))
    shared = sum(p in real_pairs for p in zip(edges["pre"], edges["post"], strict=True))
    direct = edges["pre"].isin(eye) & edges["post"].isin(dn)
    signs = neurons["sign"].to_numpy()
    return {
        "direct_eye_to_motor_connections": int(direct.sum()),
        "eye_to_motor_synapses_median": float(np.median(reached)) if len(reached) else None,
        "eye_to_motor_synapses_histogram": {
            str(int(h)): int(c)
            for h, c in zip(*np.unique(reached, return_counts=True), strict=True)
        },
        "motor_neurons_unreachable_from_eye": int((~np.isfinite(hops)).sum()),
        "connections_shared_with_real": round(shared / len(edges), 4),
        "excitatory_neuron_fraction": round(float((signs > 0).mean()), 4),
    }


def edge_sample(neurons: pd.DataFrame, edges: pd.DataFrame, per_block: int) -> list[list[int]]:
    """The strongest `per_block` connections of each layer-to-layer block, as
    [pre, post, weight, sign] (neuron indices), for drawing the wiring in 3D."""
    from brain.connectome.variants import neuron_layers

    layer = neuron_layers(neurons).to_numpy()
    e = edges.assign(block=layer[edges["pre"]] + ">" + layer[edges["post"]])
    top = (
        e.sort_values(["weight", "pre", "post"], ascending=[False, True, True], kind="stable")
        .groupby("block", sort=True)
        .head(per_block)
        .sort_values(["pre", "post"])
    )
    rows: list[list[int]] = top[["pre", "post", "weight", "sign"]].astype(int).to_numpy().tolist()
    return rows
