"""Brain data for the front end, the fly eye, and the Brain Explorer (spec 5.5).

Owner: Person 1. Format is documented in docs/PROTOCOL.md section 4.
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


def column_angles(
    columns: pd.DataFrame, spacing_deg: float, binocular_overlap_deg: float
) -> pd.DataFrame:
    """Viewing direction of each eye column, in the eye convention of fly_saber/eye.py.

    Azimuth is positive to the fly's right, elevation positive up, both in degrees.
    hex1 and hex2 are axial hex coordinates with axes 120 degrees apart: +hex1
    points up and forward, +hex2 up and backward (fit to lamina soma positions,
    see docs/DATA_NOTES.md). Each eye's most frontal column sits at
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
    """Lesionable sections and mood sliders for the UI (a brain.model.controls.ControlsConfig)."""
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
                "min": -1.0,
                "max": 1.0,
                "default": 0.0,
            }
            for m in controls.moods
        ],
        "stimulation": {"min": 0.0, "max": controls.max_stimulation},
        "noise": {"min": 0.0, "max": controls.max_noise, "default": 0.0},
    }
