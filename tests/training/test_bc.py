"""Behavior cloning against the scripted expert, with a linear policy as the brain."""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from fly_saber.config import load_config
from fly_saber.env import IDLE_CELL, NO_CUT, SaberConfig, SaberEnv
from fly_saber.eye import EyeConfig
from fly_saber.maps.dataset import load_map_set
from training.bc import BCConfig, action_heads, make_steps

ENV = SaberEnv.create(load_config(SaberConfig, "saber"), load_config(EyeConfig, "eye"))
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


def setup(tmp_path: Path, **overrides: Any) -> tuple[Any, Any]:
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
    song = {"split": "train", "song": {"duration_sec": 3.0}, "difficulty": {"note_speed_mps": 10.0}}
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
    }
    cfg = BCConfig(**{**values, **overrides})
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
