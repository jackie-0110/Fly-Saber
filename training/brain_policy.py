"""The connectome brain as a training policy, with the env's eye built from its columns.

Owner: Person 2. See docs/PROTOCOL.md sections 2 and 5.
"""

from dataclasses import dataclass

import jax
import numpy as np
import pandas as pd

from brain.config import ModelConfig
from brain.model.brain import Params, StepFn, init_params, init_state, load_graph, make_step
from brain.model.interventions import Interventions, no_interventions
from fly_saber.env import IDLE_CELL, LUMINANCE, NO_CUT, SaberConfig, SaberEnv
from fly_saber.eye import EyeConfig, column_directions


@dataclass(frozen=True)
class BrainPolicy:
    """One brain frame per env step, fed the eye's luminance channel.

    Matches the `Policy` interface of training/bc.py; the carry is the membrane potentials.
    """

    step: StepFn
    n_neurons: int
    interventions: Interventions

    def init_carry(self) -> jax.Array:
        return init_state(self.n_neurons)

    def apply(
        self, params: Params, carry: jax.Array, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        v, out = self.step(params, carry, obs[LUMINANCE], self.interventions, key)
        return v, out.logits


def eye_directions(columns: pd.DataFrame, eye_cfg: EyeConfig) -> np.ndarray:
    """Viewing directions for the brain's eye column table, in its row order."""
    return column_directions(
        columns["side"].to_numpy(), columns["hex1"].to_numpy(), columns["hex2"].to_numpy(), eye_cfg
    )


def build(
    model_cfg: ModelConfig, saber_cfg: SaberConfig, eye_cfg: EyeConfig, key: jax.Array
) -> tuple[SaberEnv, BrainPolicy, Params]:
    """Load the subgraph; return an env seeing through its eye columns, the policy, new params."""
    env_heads = [IDLE_CELL + 1, NO_CUT + 1] * saber_cfg.n_sabers
    if list(model_cfg.action_heads) != env_heads:
        raise ValueError(f"model action_heads {model_cfg.action_heads} != env heads {env_heads}")
    graph, columns, n_neurons = load_graph(model_cfg)
    env = SaberEnv.create(saber_cfg, eye_cfg, eye_directions(columns, eye_cfg))
    policy = BrainPolicy(
        make_step(graph, model_cfg, n_neurons), n_neurons, no_interventions(n_neurons)
    )
    return env, policy, init_params(key, graph, model_cfg, n_neurons)
