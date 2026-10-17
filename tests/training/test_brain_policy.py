"""The connectome brain driving the Fly Saber env through its own eye columns."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from brain.config import ModelConfig
from fly_saber.config import load_config
from fly_saber.env import SaberConfig
from fly_saber.eye import EyeConfig
from fly_saber.maps.adapt import note_arrays
from training.brain_policy import build

SABER = load_config(SaberConfig, "saber")
EYE = load_config(EyeConfig, "eye")


def model_config(subgraph_dir: Path, action_heads: list[int]) -> ModelConfig:
    return ModelConfig(
        subgraph_dir=str(subgraph_dir),
        frame_ms=33.333,
        substeps=4,
        tau_init_ms=20.0,
        tau_min_ms=10.0,
        bias_init=0.5,
        activation="relu",
        in_weight_scale=0.9,
        input_scale=1.0,
        action_heads=action_heads,
    )


def write_subgraph(directory: Path) -> None:
    """One R1-R6 column per eye and an R7 in the left one, all exciting one descending neuron."""
    pd.DataFrame(
        {
            "type": ["R1-R6", "R1-R6", "R7p", "DNp01"],
            "role": ["columnar", "columnar", "columnar", "descending"],
            "side": ["L", "R", "L", "L"],
            "hex1": [2.0, 2.0, 2.0, np.nan],
            "hex2": [4.0, 4.0, 4.0, np.nan],
        }
    ).to_parquet(directory / "neurons.parquet")
    pd.DataFrame(
        {"pre": [0, 1, 2], "post": [3, 3, 3], "weight": [10, 10, 10], "sign": [1, 1, 1]}
    ).to_parquet(directory / "edges.parquet")


def test_brain_reads_both_eye_channels(tmp_path: Path) -> None:
    write_subgraph(tmp_path)
    _, policy, params = build(
        model_config(tmp_path, [13, 5, 13, 5]), SABER, EYE, jax.random.PRNGKey(0)
    )
    params = params._replace(readout_w=jnp.ones_like(params.readout_w))
    key = jax.random.PRNGKey(1)
    apply = jax.jit(policy.apply)

    def logits_after(luminance: float, color: float) -> jax.Array:
        obs = jnp.stack([jnp.full(2, luminance), jnp.full(2, color)])
        v = policy.init_carry()
        for _ in range(30):
            v, logits = apply(params, v, obs, key)
        return jnp.asarray(logits)

    assert not jnp.allclose(logits_after(0.0, 0.5), logits_after(1.0, 0.5))
    assert not jnp.allclose(logits_after(0.5, 0.0), logits_after(0.5, 1.0))


def test_env_observes_one_value_per_brain_column_and_runs_under_vmap(tmp_path: Path) -> None:
    write_subgraph(tmp_path)
    env, policy, params = build(
        model_config(tmp_path, [13, 5, 13, 5]), SABER, EYE, jax.random.PRNGKey(0)
    )
    notes = note_arrays([], SABER.max_visible_notes)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), notes)
    obs, state = jax.vmap(env.reset, in_axes=(0, None, None, None))(
        batch, jnp.float32(8.0), jnp.float32(0.0), jnp.float32(5.0)
    )
    carry = jnp.stack([policy.init_carry()] * 2)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    carry, logits = jax.jit(jax.vmap(policy.apply, in_axes=(None, 0, 0, 0)))(
        params, carry, obs, keys
    )
    assert obs.shape == (2, 2, 2)
    assert logits.shape == (2, 36)


def test_action_heads_must_match_the_env(tmp_path: Path) -> None:
    write_subgraph(tmp_path)
    single = dataclasses.replace(SABER, two_sabers=False)
    with pytest.raises(ValueError, match="action_heads"):
        build(model_config(tmp_path, [13, 5, 13, 5]), single, EYE, jax.random.PRNGKey(0))
