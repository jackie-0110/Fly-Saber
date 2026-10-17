"""The connectome brain as a training policy, with the env's eye built from its columns.

Owner: Person 2. See docs/PROTOCOL.md sections 2 and 5.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
import pandas as pd

from brain.config import ModelConfig
from brain.connectome.metadata import column_angles
from brain.model.brain import (
    Params,
    StepFn,
    init_params,
    init_state,
    load_graph,
    make_step,
    rest_state,
    with_live_baseline,
)
from brain.model.interventions import Interventions, no_interventions
from fly_saber.env import IDLE_CELL, NO_CUT, SaberConfig, SaberEnv
from fly_saber.eye import EyeConfig
from training.calibration import fit_readout


@dataclass(frozen=True)
class BrainPolicy:
    """One brain frame per env step, fed the eye's luminance and color channels.

    Matches the `Policy` interface of training/bc.py; the carry is the membrane potentials.
    """

    step: StepFn
    n_neurons: int
    interventions: Interventions
    rest: jax.Array | None = None
    baseline: Callable[[Params], Params] | None = None
    """Sets the live readout baseline from the current params (with_live_baseline)."""

    def prepare(self, params: Params) -> Params:
        """Params to run with: call once per update on the trained params (training/bc.py)."""
        return params if self.baseline is None else self.baseline(params)

    """Episode start state (brain.model.brain.rest_state); None starts from all zeros.
    After loading a checkpoint, replace it: dataclasses.replace(policy, rest=rest_state(...))."""

    def init_carry(self) -> jax.Array:
        return init_state(self.n_neurons) if self.rest is None else self.rest

    def apply(
        self, params: Params, carry: jax.Array, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        v, out = self.step(params, carry, obs, self.interventions, key)
        return v, out.logits


def eye_directions(columns: pd.DataFrame, eye_cfg: EyeConfig) -> np.ndarray:
    """(n_columns, 2) radians (azimuth, elevation) of the brain's eye columns, in row order."""
    angles = column_angles(columns, eye_cfg.ommatidium_spacing_deg, eye_cfg.binocular_overlap_deg)
    radians: np.ndarray = np.radians(angles[["azimuth_deg", "elevation_deg"]].to_numpy())
    return radians


def build(
    model_cfg: ModelConfig, saber_cfg: SaberConfig, eye_cfg: EyeConfig, key: jax.Array
) -> tuple[SaberEnv, BrainPolicy, Params]:
    """Load the subgraph; return an env seeing through its eye columns, the policy, new params."""
    env_heads = [IDLE_CELL + 1, NO_CUT + 1] * saber_cfg.n_sabers
    if list(model_cfg.action_heads) != env_heads:
        raise ValueError(f"model action_heads {model_cfg.action_heads} != env heads {env_heads}")
    graph, columns, n_neurons = load_graph(model_cfg)
    env = SaberEnv.create(saber_cfg, eye_cfg, eye_directions(columns, eye_cfg))
    k_params, k_cal = jax.random.split(key)
    params = init_params(k_params, graph, model_cfg, n_neurons)
    if model_cfg.readout_norm:  # fit the readout on expert play (Person 1, training/calibration.py)
        params, stats = fit_readout(env, params, graph, model_cfg, n_neurons, k_cal)
        print(f"readout fit on expert play: {stats}", flush=True)
    rest = rest_state(params, graph, model_cfg, n_neurons)
    baseline = None
    if model_cfg.readout_norm and model_cfg.live_baseline:

        def baseline(p: Params) -> Params:
            return with_live_baseline(p, graph, model_cfg, n_neurons, rest)

        params = baseline(params)
    policy = BrainPolicy(
        make_step(graph, model_cfg, n_neurons),
        n_neurons,
        no_interventions(n_neurons),
        rest,
        baseline,
    )
    return env, policy, params
