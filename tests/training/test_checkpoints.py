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
