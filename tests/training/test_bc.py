"""Behavior cloning against the scripted expert, with a linear policy as the brain."""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from brain.model.brain import init_params as init_brain_params
from brain.model.brain import load_graph
from fly_saber.config import load_config
from fly_saber.env import IDLE_CELL, NO_CUT, SaberConfig, SaberEnv
from fly_saber.eye import EyeConfig
from fly_saber.maps.dataset import load_map_set
from tests.training.test_brain_policy import model_config, write_subgraph
from training.bc import BCConfig, action_heads, make_steps, run
from training.checkpoints import load_checkpoint, subgraph_fingerprint

SABER = load_config(SaberConfig, "saber")
EYE = load_config(EyeConfig, "eye")
ENV = SaberEnv.create(SABER, EYE)
HEADS = action_heads(ENV.cfg.n_sabers)


class LinearPolicy:
    def init_carry(self) -> jax.Array:
        return jnp.zeros(())

    def apply(
        self, params: Any, carry: jax.Array, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return carry, (obs.reshape(-1) - 0.5) @ params["w"] + params["b"]


def init_params(bias: jax.Array | None = None) -> dict[str, jax.Array]:
    n_obs = 2 * ENV.directions.shape[0]
    return {
        "w": jnp.zeros((n_obs, sum(HEADS))),
        "b": jnp.zeros(sum(HEADS)) if bias is None else bias,
    }


def bc_config(tmp_path: Path, **overrides: Any) -> BCConfig:
    """Write a one-map fixture into `tmp_path` and a config that trains on it."""
    notes = [
        {
            "id": i,
            "time_sec": 0.4 + 0.3 * i,
            "lane": i % 4,
            "layer": i % 3,
            "color": i % 2,
            "cut_direction": i % 4,
            "is_bomb": False,
        }
        for i in range(6)
    ]
    song = {"split": "train", "song": {"duration_sec": 3.0}}
    (tmp_path / "abc_easy.json").write_text(
        json.dumps({"map_id": "abc_easy", "notes": notes, **song})
    )
    values = {
        "maps_dir": str(tmp_path),
        "split": "train",
        "min_gap_sec": 0.0,
        "window_sec": 3.0,
        "batch_size": 4,
        "learning_rate": 0.003,
        "n_updates": 1,
        "log_every": 1,
        "seed": 0,
        "variant": "test",
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    return BCConfig(**{**values, **overrides})


def setup(tmp_path: Path) -> tuple[Any, Any]:
    cfg = bc_config(tmp_path)
    map_set, _ = load_map_set(tmp_path, "train", ENV.cfg.max_visible_notes)
    optimizer = optax.adam(cfg.learning_rate)
    return make_steps(ENV, LinearPolicy(), map_set, cfg, optimizer), optimizer


def test_bc_lowers_the_loss(tmp_path: Path) -> None:
    (update, _), optimizer = setup(tmp_path)
    params = init_params()
    opt_state = optimizer.init(params)
    losses = []
    for i in range(40):
        params, opt_state, metrics = update(params, opt_state, jax.random.PRNGKey(i))
        losses.append(float(metrics.loss))
    assert losses[-1] < 0.5 * losses[0]


def test_evaluate_moves_the_env_with_the_policy(tmp_path: Path) -> None:
    (_, evaluate), _ = setup(tmp_path)
    idle = jnp.zeros(sum(HEADS)).at[
        jnp.array([IDLE_CELL, 13 + NO_CUT, 18 + IDLE_CELL, 31 + NO_CUT])
    ]
    played = evaluate(init_params(idle.set(10.0)), jax.random.PRNGKey(0))
    assert int(played.hits) == 0
    assert int(played.misses) == 4 * 6


def test_run_saves_a_brain_checkpoint_that_loads_back(tmp_path: Path) -> None:
    subgraph = tmp_path / "subgraph"
    subgraph.mkdir()
    write_subgraph(subgraph)
    counts = {"n_neurons": 4, "n_edges": 3, "n_synapses": 30}
    (subgraph / "meta.json").write_text(json.dumps(counts))
    model_cfg = model_config(subgraph, action_heads(SABER.n_sabers))

    path = run(bc_config(tmp_path, n_updates=2), model_cfg, SABER, EYE)

    graph, _, n_neurons = load_graph(model_cfg)
    template = init_brain_params(jax.random.PRNGKey(1), graph, model_cfg, n_neurons)
    _, meta = load_checkpoint(path, template, expect_subgraph=subgraph_fingerprint(subgraph))
    run_dir = tmp_path / "checkpoints" / "test" / "default" / "seed0"
    assert path == run_dir / "bc_2.npz"
    assert (meta["stage"], meta["step"]) == ("bc", 2)
    assert (run_dir / "config.json").exists() and (run_dir / "bc_log.json").exists()


def test_run_checkpoints_periodically_into_its_run_dir(tmp_path: Path) -> None:
    import dataclasses

    subgraph = tmp_path / "subgraph"
    subgraph.mkdir()
    write_subgraph(subgraph)
    (subgraph / "meta.json").write_text(
        json.dumps({"n_neurons": 4, "n_edges": 3, "n_synapses": 30})
    )
    model_cfg = model_config(subgraph, action_heads(SABER.n_sabers))
    cfg = dataclasses.replace(
        bc_config(tmp_path, n_updates=4), run_name="exp1", checkpoint_every=2, log_every=2
    )
    run(cfg, model_cfg, SABER, EYE)
    run_dir = tmp_path / "checkpoints" / "test" / "exp1" / "seed0"
    assert sorted(p.name for p in run_dir.glob("bc_*.npz")) == ["bc_2.npz", "bc_4.npz"]
    assert [row["update"] for row in json.loads((run_dir / "bc_log.json").read_text())] == [2, 4]


def test_apply_overrides() -> None:
    import pytest

    from brain.config import ModelConfig
    from training.bc import BCConfig, apply_overrides

    base: dict[str, Any] = {
        "train": BCConfig("", "train", 0.0, 1.0, 1, 1e-2, 1, 1, 0, "real", ""),
        "model": ModelConfig("", 33.3, 4, 20.0, 10.0, 0.5, "relu", 0.9, 1.0, [13, 5]),
    }
    out = apply_overrides(
        base,
        [
            "run_name=fast",
            "window_sec=5",
            "batch_size=256",
            "brain_learning_rate=1e-4",
            "model.freeze_brain=true",
            "model.subgraph_dir=data/x",
        ],
    )
    t, m = out["train"], out["model"]
    assert (t.run_name, t.window_sec, t.batch_size, t.brain_learning_rate) == (
        "fast",
        5.0,
        256,
        1e-4,
    )
    assert m.freeze_brain is True and m.subgraph_dir == "data/x"
    assert base["train"].run_name == "default"  # originals untouched
    for bad in (
        ["nope=1"],
        ["model.nope=1"],
        ["x.y=1"],
        ["batch_size"],
        ["model.freeze_brain=yes"],
    ):
        with pytest.raises(ValueError):
            apply_overrides(base, bad)


def test_optimizer_never_moves_fixed_brain_buffers() -> None:
    import dataclasses

    import optax

    from brain.model.brain import Params
    from training.bc import BCConfig, make_optimizer

    zeros = jnp.zeros(3)
    params = Params(*([zeros] * len(Params._fields)))
    grads = Params(*([jnp.ones(3)] * len(Params._fields)))
    base = BCConfig("", "train", 0.0, 1.0, 1, 1e-2, 1, 1, 0, "real", "")
    opt = make_optimizer(dataclasses.replace(base, brain_learning_rate=1e-3), params)
    updates, _ = opt.update(grads, opt.init(params), params)
    new = optax.apply_updates(params, updates)
    assert float(jnp.abs(new.dn_mean).max()) == 0.0 and float(jnp.abs(new.dn_std).max()) == 0.0
    assert float(jnp.abs(new.bias).max()) > 0.0 and float(jnp.abs(new.readout_w).max()) > 0.0
