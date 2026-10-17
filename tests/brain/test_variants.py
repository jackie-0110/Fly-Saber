"""Control brains keep degrees and signs as promised."""

import numpy as np
import pandas as pd

from brain.connectome.variants import rewire, shuffle_signs


def _graph(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n, e = 60, 400
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < e:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((int(a), int(b)))
    pre, post = np.array(sorted(pairs)).T
    sign = rng.choice([-1, 1], n)
    neurons = pd.DataFrame({"sign": sign, "nt": np.where(sign > 0, "acetylcholine", "gaba")})
    edges = pd.DataFrame(
        {"pre": pre, "post": post, "weight": rng.integers(5, 50, e), "sign": sign[pre]}
    )
    return neurons, edges


def test_rewire_preserves_degrees_and_weights() -> None:
    _, edges = _graph()
    out = rewire(edges, np.random.default_rng(1))
    for col in ("pre", "post"):
        assert (
            out[col].value_counts().sort_index() == edges[col].value_counts().sort_index()
        ).all()
    assert len(out) == len(edges) and sorted(out["weight"]) == sorted(edges["weight"])
    assert not (out["pre"] == out["post"]).any() and not out.duplicated(["pre", "post"]).any()
    # Weight and sign stay with the presynaptic side.
    for p in edges["pre"].unique():
        assert sorted(out.loc[out["pre"] == p, "weight"]) == sorted(
            edges.loc[edges["pre"] == p, "weight"]
        )
    pre_sign = edges.groupby("pre")["sign"].first()
    assert (out["sign"].to_numpy() == pre_sign.reindex(out["pre"]).to_numpy()).all()
    before = set(zip(edges["pre"], edges["post"], strict=True))
    overlap = len(set(zip(out["pre"], out["post"], strict=True)) & before)
    assert overlap < len(edges) * 0.2  # actually rewired


def test_rewire_is_seeded() -> None:
    _, edges = _graph()
    a, b = rewire(edges, np.random.default_rng(3)), rewire(edges, np.random.default_rng(3))
    assert a.equals(b)


def test_shuffle_signs_keeps_wiring_and_sign_counts() -> None:
    neurons, edges = _graph()
    n2, e2 = shuffle_signs(neurons, edges, np.random.default_rng(2))
    assert (e2[["pre", "post", "weight"]] == edges[["pre", "post", "weight"]]).all().all()
    assert sorted(n2["sign"]) == sorted(neurons["sign"])
    assert (e2["sign"].to_numpy() == n2["sign"].to_numpy()[e2["pre"]]).all()
    assert (n2["sign"] != neurons["sign"]).any()


def test_rewire_layered_keeps_layer_blocks_and_degrees() -> None:
    from brain.connectome.variants import neuron_layers, rewire_layered

    rng = np.random.default_rng(0)
    types = ["R1-R6"] * 10 + ["L1"] * 10 + ["Mi1"] * 10 + ["LC4"] * 10 + ["DNp01"] * 10
    roles = ["columnar"] * 30 + ["projection"] * 10 + ["descending"] * 10
    neurons = pd.DataFrame({"type": types, "role": roles})
    # Strictly layered: each layer only feeds the next.
    rows = []
    for a in range(4):
        for i in range(10 * a, 10 * a + 10):
            for j in rng.choice(np.arange(10 * (a + 1), 10 * (a + 2)), 4, replace=False):
                rows.append((i, int(j)))
    pre, post = np.array(rows).T
    edges = pd.DataFrame(
        {"pre": pre, "post": post, "weight": rng.integers(5, 50, len(pre)), "sign": 1}
    )
    out, stats = rewire_layered(edges, neurons, np.random.default_rng(1))
    layer = neuron_layers(neurons).to_numpy()

    def block(e: pd.DataFrame) -> list[tuple[str, str]]:
        return sorted(zip(layer[e["pre"]], layer[e["post"]], strict=True))

    assert block(out) == block(edges)  # same layer-to-layer counts: nothing skips a layer
    for col in ("pre", "post"):
        assert (
            out[col].value_counts().sort_index() == edges[col].value_counts().sort_index()
        ).all()
    assert not out.duplicated(["pre", "post"]).any() and stats["kept_edges"] == 0
    shared = set(zip(out["pre"], out["post"], strict=True)) & set(zip(pre, post, strict=True))
    assert len(shared) < len(edges) * 0.7  # actually shuffled within blocks
