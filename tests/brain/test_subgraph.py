"""Subgraph extraction on a tiny hand-built connectome."""

import numpy as np
import pandas as pd
import pytest

from brain.config import SubgraphConfig
from brain.connectome.groups import build_groups
from brain.connectome.signs import nt_to_sign
from brain.connectome.subgraph import build_subgraph, inherit_hex, match_types

NAN = np.nan


def _cfg(**overrides: object) -> SubgraphConfig:
    base: dict[str, object] = dict(
        dataset="toy",
        raw_dir="",
        out_dir="",
        min_synapses=5,
        column_stride=2,
        columnar_types=["R1-R6", "L[1-5]", "T4[a-d]"],
        projection_types=["LPLC2"],
        descending_superclass="descending_neuron",
        max_intermediates=10,
        max_neurons=100,
        hex_inherit_superclasses=["ol_intrinsic", "ol_sensory"],
        hex_inherit_passes=2,
    )
    base.update(overrides)
    return SubgraphConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Column (2, 2) is sampled with stride 2; column (1, 1) is not.
    rows = [
        # body, type, superclass, side, hex1, hex2, nt, x
        (1, "R1-R6", "ol_sensory", "L", NAN, NAN, "histamine", NAN),
        (2, "L1", "ol_intrinsic", "L", 2, 2, "glutamate", 10.0),
        (3, "T4a", "ol_intrinsic", "L", NAN, NAN, "acetylcholine", 20.0),
        (4, "L1", "ol_intrinsic", "L", 1, 1, "glutamate", 30.0),
        (5, "LPLC2", "visual_projection", "L", NAN, NAN, "acetylcholine", 40.0),
        (6, "PVLP011", "cb_intrinsic", "L", NAN, NAN, "acetylcholine", 50.0),
        (7, "DNp01", "descending_neuron", "L", NAN, NAN, "acetylcholine", 60.0),
        (8, "Other", "cb_intrinsic", "L", NAN, NAN, "gaba", 70.0),
    ]
    cols = ["body_id", "type", "superclass", "side", "hex1", "hex2", "nt", "x"]
    neurons = pd.DataFrame(rows, columns=cols).set_index("body_id")
    neurons["y"] = neurons["x"]
    neurons["z"] = neurons["x"]
    edges = pd.DataFrame(
        [(1, 2, 50), (2, 3, 20), (3, 5, 30), (4, 5, 5), (5, 6, 40), (6, 7, 25), (8, 7, 9)],
        columns=["body_pre", "body_post", "weight"],
    )
    return neurons, edges


def test_signs() -> None:
    nts = pd.Series(["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "mystery"])
    assert nt_to_sign(nts).tolist() == [1, -1, -1, -1, 0, 0]


def test_match_types_is_full_match() -> None:
    types = pd.Series(["T4a", "T4", "LT4a", "R1-R6"])
    assert match_types(types, ["T4[a-d]", "R1-R6"]).tolist() == [True, False, False, True]


def test_inherit_hex_from_strongest_partner(toy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    neurons, edges = toy
    out = inherit_hex(neurons, edges, ["ol_intrinsic", "ol_sensory"], passes=2)
    assert (out.loc[1, "hex1"], out.loc[1, "hex2"]) == (2, 2)  # photoreceptor from L1
    assert (out.loc[3, "hex1"], out.loc[3, "hex2"]) == (2, 2)  # T4 from L1
    assert np.isnan(out.loc[5, "hex1"])  # projection neurons never inherit


def test_build_subgraph(toy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    neurons, edges = toy
    sub = build_subgraph(neurons, edges, _cfg())
    kept = dict(zip(sub.neurons["body_id"], sub.neurons["role"], strict=True))
    assert kept == {
        1: "columnar",
        2: "columnar",
        3: "columnar",
        5: "projection",
        7: "descending",
        6: "intermediate",
    }  # 4 is in an unsampled column; 8 gets no projection input
    assert list(sub.neurons.index) == list(range(len(sub.neurons)))
    assert sub.neurons.loc[sub.neurons["body_id"] == 1, "position_inherited"].item()

    body = sub.neurons["body_id"].to_numpy()
    pairs = {(body[p], body[q]) for p, q in zip(sub.edges["pre"], sub.edges["post"], strict=True)}
    assert pairs == {(1, 2), (2, 3), (3, 5), (5, 6), (6, 7)}
    sign_of = dict(zip(sub.neurons.index, sub.neurons["sign"], strict=True))
    assert all(sign_of[p] == s for p, s in zip(sub.edges["pre"], sub.edges["sign"], strict=True))


def test_budget_never_prunes_seeds(toy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    neurons, edges = toy
    sub = build_subgraph(neurons, edges, _cfg(max_neurons=5))
    assert "intermediate" not in set(sub.neurons["role"])
    assert len(sub.neurons) == 5


def test_groups(toy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    neurons, edges = toy
    sub = build_subgraph(neurons, edges, _cfg())
    groups = build_groups(sub.neurons)
    idx = {b: i for i, b in enumerate(sub.neurons["body_id"])}
    assert groups["giant_fiber"] == [idx[7]]
    assert groups["DN_all"] == [idx[7]]
    assert groups["photoreceptors_left"] == [idx[1]]
    assert "photoreceptors_right" not in groups


def test_modulatory_neurons(toy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    neurons, edges = toy
    extra = pd.DataFrame(
        [
            (9, "OA-VUMa1", "cb_intrinsic", "M", NAN, NAN, "octopamine", 80.0),
            (10, "PPM1201", "cb_intrinsic", "L", NAN, NAN, "dopamine", 90.0),
            (11, "PAM01", "cb_intrinsic", "L", NAN, NAN, "dopamine", 95.0),
        ],
        columns=["body_id", "type", "superclass", "side", "hex1", "hex2", "nt", "x"],
    ).set_index("body_id")
    extra["y"] = extra["z"] = extra["x"]
    neurons = pd.concat([neurons, extra])
    edges = pd.concat(
        [
            edges,
            pd.DataFrame(
                [(9, 5, 30), (10, 7, 25), (11, 7, 6)], columns=["body_pre", "body_post", "weight"]
            ),
        ]
    )
    cfg = _cfg(
        modulatory_nts=["octopamine", "dopamine"],
        max_modulatory_per_nt=5,
        min_modulatory_synapses=20,
    )
    sub = build_subgraph(neurons, edges, cfg)
    role = dict(zip(sub.neurons["body_id"], sub.neurons["role"], strict=True))
    assert role[9] == "modulatory" and role[10] == "modulatory"
    assert 11 not in role  # 6 synapses onto the subgraph, below min_modulatory_synapses
    assert list(sub.neurons["role"]).index("modulatory") > list(sub.neurons["role"]).index(
        "intermediate"
    )
    groups = build_groups(sub.neurons)
    idx = {b: i for i, b in enumerate(sub.neurons["body_id"])}
    assert groups["octopamine"] == [idx[9]] and groups["dopamine"] == [idx[10]]
