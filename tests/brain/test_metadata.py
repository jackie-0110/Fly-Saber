"""Front end neuron export."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from brain.connectome.metadata import web_explorer, web_neurons


def test_web_explorer_describes_subtypes_and_rejects_unknown_types() -> None:
    entries = [
        {"name": "T4", "types": ["T4a", "T4b"], "description": "Direction\n  selective."},
        {"name": "Giant fiber", "types": ["DNp01"], "description": "Escape."},
    ]
    out = web_explorer(["DNp01", "T4a", "T4b", "Tm3"], entries)
    assert out["descriptions"][0] == {
        "name": "T4",
        "types": ["T4a", "T4b"],
        "description": "Direction selective.",
    }
    with pytest.raises(ValueError, match="not in the subgraph"):
        web_explorer(["DNp01"], entries)
    with pytest.raises(ValueError, match="described twice"):
        web_explorer(["DNp01", "T4a", "T4b"], [*entries, {**entries[1], "name": "GF"}])


def test_committed_explorer_matches_config_and_covers_the_spec_types() -> None:
    """The exported explorer.json is current, and has the 15 types spec 5.5 names."""
    types = json.loads(Path("frontend/public/data/brain/neurons.json").read_text())["cell_types"]
    entries = yaml.safe_load(Path("configs/cell_types.yaml").read_text())["entries"]
    committed = json.loads(Path("frontend/public/data/brain/explorer.json").read_text())
    assert committed == web_explorer(types, entries)
    described = {t for d in committed["descriptions"] for t in d["types"]}
    spec = ["T4a", "T5a", "L1", "L2", "Mi1", "Tm3", "LPLC2", "LC4", "LC11", "HSE", "VS"]
    spec += ["DNp01", "R1-R6", "R7p", "R8p"]
    assert set(spec) <= described


def test_web_neurons() -> None:
    neurons = pd.DataFrame(
        {
            "body_id": [11, 12, 13, 14],
            "type": ["T4a", "LPLC2", "T4a", "DNp01"],
            "superclass": [
                "ol_intrinsic",
                "visual_projection",
                "ol_intrinsic",
                "descending_neuron",
            ],
            "role": ["columnar", "projection", "columnar", "descending"],
            "side": ["L", "R", None, "L"],
            "nt": ["acetylcholine"] * 4,
            "sign": [1, 1, 1, 1],
            "x": [1000.0, 2000.0, np.nan, 4000.0],
            "y": [0.0, 0.0, np.nan, 0.0],
            "z": [0.0, 0.0, np.nan, 0.0],
            "position_inherited": [False, True, False, False],
        }
    )
    out = web_neurons(neurons)
    assert out["n_neurons"] == 4
    assert out["cell_types"] == ["DNp01", "LPLC2", "T4a"]
    assert [out["cell_types"][t] for t in out["type"]] == ["T4a", "LPLC2", "T4a", "DNp01"]
    assert [out["roles"][r] for r in out["role"]] == neurons["role"].tolist()
    assert [out["sides"][s] for s in out["side"]] == ["L", "R", "M", "L"]
    # Neuron 2 has no position: it takes its type's mean (neuron 0) and is flagged.
    assert out["position_um"][6:9] == out["position_um"][0:3] == [8.0, 0.0, 0.0]
    sources = [out["position_sources"][s] for s in out["position_source"]]
    assert sources == ["soma", "partner", "type_mean", "soma"]
    assert len(out["position_um"]) == 12 and len(out["body_id"]) == 4


def test_column_angles_orientation() -> None:
    from brain.connectome.metadata import column_angles

    cols = pd.DataFrame(
        {"side": ["L", "L", "L", "R", "R", "R"], "hex1": [2, 4, 2] * 2, "hex2": [2, 2, 4] * 2}
    )
    a = column_angles(cols, spacing_deg=5.0, binocular_overlap_deg=10.0).round(2)
    # +hex1 is forward: (4, 2) is the left eye's most frontal column, 10 deg past the midline.
    assert a["azimuth_deg"].tolist()[:3] == [1.34, 10.0, -7.32]
    # Right eye mirrors azimuth only.
    assert a["azimuth_deg"].tolist()[3:] == [-1.34, -10.0, 7.32]
    # hex1 + hex2 is up; centered on the eye's vertical midpoint.
    assert a["elevation_deg"].tolist() == [-2.5, 2.5, 2.5] * 2


def test_web_columns_flags_fallback() -> None:
    from brain.connectome.metadata import web_columns

    cols = pd.DataFrame({"side": ["L", "L"], "hex1": [2.0, 4.0], "hex2": [2.0, 2.0]})
    inputs = pd.DataFrame(
        {"neuron": [0, 1, 2], "column": [0, 1, 1], "channel": [0, 0, 1], "sign": [1.0, -1.0, 1.0]}
    )
    out = web_columns(cols, inputs, spacing_deg=5.0, binocular_overlap_deg=10.0)
    assert out["n_columns"] == 2 and out["kept_spacing_deg"] == 10.0
    assert out["luminance_input"] == [0, 1]  # column 1 has only L1/L2 fallback for luminance
    assert out["hex1"] == [2, 4]


def _modes_brain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """R -> L1 -> Mi1 -> LC4 -> DN, plus a shortcut variant wiring R straight to the DN."""
    neurons = pd.DataFrame(
        {
            "type": ["R1-R6", "L1", "Mi1", "LC4", "DNp01"],
            "role": ["columnar", "columnar", "columnar", "projection", "descending"],
            "sign": [-1, -1, 1, 1, 1],
        }
    )
    real = pd.DataFrame(
        {"pre": [0, 1, 2, 3], "post": [1, 2, 3, 4], "weight": [9, 8, 7, 6], "sign": [-1, -1, 1, 1]}
    )
    short = pd.DataFrame(
        {"pre": [0, 1, 2, 0], "post": [1, 2, 3, 4], "weight": [9, 8, 7, 6], "sign": [-1, -1, 1, -1]}
    )
    return neurons, real, short


def test_wiring_stats() -> None:
    from brain.connectome.metadata import wiring_stats

    neurons, real, short = _modes_brain()
    r = wiring_stats(neurons, real, real)
    # Eye inputs are R and L1, so the DN is 3 synapses from L1 in the real chain.
    assert r["direct_eye_to_motor_connections"] == 0 and r["eye_to_motor_synapses_median"] == 3.0
    assert r["connections_shared_with_real"] == 1.0 and r["excitatory_neuron_fraction"] == 0.6
    s = wiring_stats(neurons, short, real)
    assert s["direct_eye_to_motor_connections"] == 1 and s["eye_to_motor_synapses_median"] == 1.0
    assert s["connections_shared_with_real"] == 0.75


def test_edge_sample_keeps_strongest_per_block() -> None:
    from brain.connectome.metadata import edge_sample

    neurons, real, _ = _modes_brain()
    extra = pd.DataFrame(
        {"pre": [2], "post": [3], "weight": [99], "sign": [1]}
    )  # same block as 2->3
    edges = pd.concat([real, extra], ignore_index=True)
    sample = edge_sample(neurons, edges, per_block=1)
    assert [2, 3, 99, 1] in sample and [2, 3, 7, 1] not in sample
    # One per block: photoreceptor>lamina, lamina>medulla, medulla>projection, projection>DN.
    assert len(sample) == 4
