"""Front end neuron export."""

import numpy as np
import pandas as pd

from brain.connectome.metadata import web_neurons


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
