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
