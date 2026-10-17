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
