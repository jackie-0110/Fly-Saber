"""Checkpoint round trip and safety checks."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from training.checkpoints import config_hash, load_checkpoint, read_meta, save_checkpoint


class P(NamedTuple):
    w: jnp.ndarray
    b: jnp.ndarray


@dataclass(frozen=True)
class Cfg:
    lr: float


def test_round_trip(tmp_path: Path) -> None:
    params = P(w=jnp.arange(6.0).reshape(2, 3), b=jnp.array(1.5))
    path = save_checkpoint(
        tmp_path / "real/seed0/bc_100",
        params,
        variant="real",
        stage="bc",
        step=100,
        configs={"train": Cfg(lr=1e-3)},
        subgraph={"n_neurons": 2},
        metrics={"hit_rate": 0.4},
    )
    assert path.suffix == ".npz"
    template = P(w=jnp.zeros((2, 3)), b=jnp.zeros(()))
    loaded, meta = load_checkpoint(path, template, expect_subgraph={"n_neurons": 2})
    np.testing.assert_array_equal(loaded.w, params.w)
    assert float(loaded.b) == 1.5
    assert meta["variant"] == "real" and meta["step"] == 100 and meta["metrics"]["hit_rate"] == 0.4
    assert meta["config_hash"] == config_hash({"train": {"lr": 1e-3}})
    assert read_meta(path)["stage"] == "bc"


def test_rejects_mismatch(tmp_path: Path) -> None:
    params = P(w=jnp.zeros((2, 3)), b=jnp.zeros(()))
    path = save_checkpoint(
        tmp_path / "c",
        params,
        variant="real",
        stage="bc",
        step=1,
        configs={},
        subgraph={"n_neurons": 2},
    )
    with pytest.raises(ValueError, match="subgraph"):
        load_checkpoint(path, params, expect_subgraph={"n_neurons": 3})
    with pytest.raises(ValueError, match="shape"):
        load_checkpoint(path, P(w=jnp.zeros((3, 3)), b=jnp.zeros(())))


def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": {"x": 1, "y": 2}}) == config_hash({"a": {"y": 2, "x": 1}})
    assert config_hash({"a": {"x": 1}}) != config_hash({"a": {"x": 2}})


def test_fingerprint_catches_reordered_neurons(tmp_path: Path) -> None:
    import json

    import pandas as pd

    from training.checkpoints import subgraph_fingerprint, subgraph_mismatch

    for name, order in (("a", [11, 12, 13]), ("b", [12, 11, 13])):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"body_id": order}).to_parquet(d / "neurons.parquet")
        pd.DataFrame({"pre": [0], "post": [1], "weight": [5]}).to_parquet(d / "edges.parquet")
        (d / "meta.json").write_text(json.dumps({"n_neurons": 3, "n_edges": 1, "n_synapses": 5}))
    a, b = subgraph_fingerprint(tmp_path / "a"), subgraph_fingerprint(tmp_path / "b")
    assert subgraph_mismatch(a, b) == ["neurons_sha256"]  # same counts, different slots
    old_style = {"n_neurons": 3, "n_edges": 1, "n_synapses": 5}
    assert subgraph_mismatch(old_style, b) == []  # older checkpoints: counts only
