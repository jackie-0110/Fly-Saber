"""The real Fly Saber env behind the session's Sim interface (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from brain.connectome.metadata import cell_types
from brain.model.brain import Params
from brain.model.controls import from_controls, load_controls
from brain.model.interventions import Interventions, no_interventions
from fly_saber.config import load_config
from fly_saber.env import Event, SaberConfig, SaberEnv, SaberState
from fly_saber.experts import saber_expert
from fly_saber.eye import EyeConfig, eye_controls
from fly_saber.maps.adapt import note_arrays
from server.protocol import RenderState
from server.session import NoteEvent, StepOutput
from training.bc import action_heads, greedy_action
from training.brain_policy import BrainPolicy
from training.play_eval import load_trained

Act = Callable[[SaberState, jax.Array, Any, Any, jax.Array], tuple[jax.Array, Any, jax.Array]]
"""(state, obs, carry, interventions, key) -> ((n_sabers, 2) action, carry, rates) for one
unbatched env. The carry is the player's memory between frames: a brain's membrane
potentials. Rates are the brain's (n_neurons,) firing rates, empty for the expert."""
Encode = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
"""Rates -> (per-neuron, per-cell-type) int8 activity payloads (PROTOCOL.md section 6)."""

EVENT_KINDS: dict[int, Literal["hit", "miss", "bad_cut", "bomb"]] = {
    Event.HIT: "hit",
    Event.MISS: "miss",
    Event.BAD_CUT: "bad_cut",
    Event.BOMB: "bomb",
}
# Past any song: the episode ends on its own after the last note.
OPEN_END_SEC = 1.0e6
# Every map pads to this one size, so the step compiles once, at startup; a new size would
# compile on the first frame, after the audio has started. Step time does not grow with it.
MAX_NOTES = 4096
# Rate change from rest that maps to int8 +-127 (PROTOCOL.md section 6). Live rates move by
# hundredths around ~0.7, so a full 0-to-rate_max scale would show a frozen brain.
ACTIVITY_SCALE = 0.25


def no_activity(rates: jax.Array) -> tuple[jax.Array, jax.Array]:
    empty = jnp.zeros(0, jnp.int8)
    return empty, empty


class LiveSaber:
    """Sim factory for one player; the env step compiles once, for `MAX_NOTES` notes."""

    def __init__(
        self,
        env: SaberEnv,
        act: Act,
        carry: Any,
        interventions: Callable[[Mapping[str, Any]], Any] = lambda controls: (),
        n_neurons: int = 0,
        neuron_cell_types: Sequence[str] = (),
        encode: Encode = no_activity,
    ) -> None:
        """`interventions` turns set_interventions brain controls (PROTOCOL.md section 4)
        into what `act` takes; it raises ValueError on bad controls."""
        self.env = env
        self.carry = carry
        self.interventions = interventions
        self.n_neurons = n_neurons
        self.cell_types = list(neuron_cell_types)
        self.reset_fn = jax.jit(env.reset)

        def step(state: SaberState, obs: jax.Array, carry: Any, iv: Any, key: jax.Array) -> Any:
            key, k_act = jax.random.split(key)
            action, carry, rates = act(state, obs, carry, iv, k_act)
            obs, state, _, done, events = env.step(state, action)
            return state, obs, carry, key, done, events, encode(rates)

        self.step_fn = jax.jit(step)

    def __call__(self, map_data: dict[str, Any], seed: int) -> "EnvSaberSim":
        return EnvSaberSim(self, map_data, seed)


def expert_sims() -> LiveSaber:
    """The real env played by the scripted expert."""
    env = SaberEnv.create(load_config(SaberConfig, "saber"), load_config(EyeConfig, "eye"))
    return LiveSaber(
        env,
        lambda state, obs, carry, iv, key: (saber_expert(env, state), carry, jnp.zeros(0)),
        (),
    )


def brain_sims(checkpoint: Path, subgraph_dir: Path) -> LiveSaber:
    """The env played greedily by a trained brain, seeing through its eye columns.

    `subgraph_dir` must hold the exact subgraph it was trained on (a bundle's subgraph/).
    """
    trained = load_trained(checkpoint, subgraph_dir)
    policy, base_params = trained.policy, trained.params
    heads = action_heads(trained.env.cfg.n_sabers)

    controls = load_controls()
    # Re-center the readout on the modified brain's rest (brain.model.brain.rest_dn). The
    # readout reads DN ripples of about 1e-5; without this every lesion or mood shifts the
    # DN background far more and collapses the trained fly to ~0% hits. Compiled once;
    # each later change costs about 45 brain frames.
    recenter = jax.jit(lambda iv: policy.prepare(base_params, iv))

    def act(
        state: SaberState,
        obs: jax.Array,
        v: jax.Array,
        controlled: tuple[Interventions, Params],
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        iv, params = controlled
        v, out = policy.step(params, v, obs, iv, key)
        return greedy_action(out.logits, heads), v, out.rates

    def interventions(ui: Mapping[str, Any]) -> tuple[Interventions, Params]:
        """Brain controls and the params re-centered for them; both are step arguments,
        so changing controls never recompiles."""
        iv = from_controls(ui, trained.groups, controls, trained.n_neurons)
        return iv, recenter(iv)

    neurons = pd.read_parquet(subgraph_dir / "neurons.parquet", columns=["type"])
    types = cell_types(neurons)
    return LiveSaber(
        trained.env,
        act,
        policy.init_carry(),
        interventions,
        trained.n_neurons,
        types,
        activity_encoder(policy, base_params, trained.env, neurons["type"], types),
    )


def activity_encoder(
    policy: BrainPolicy, params: Params, env: SaberEnv, neuron_types: pd.Series, types: list[str]
) -> Encode:
    """Activity as change from the untouched brain's rest on the gray background, so lesioned
    neurons read below zero."""
    gray = jnp.full((2, env.directions.shape[0]), 0.5)
    iv = no_interventions(policy.n_neurons)
    rest = policy.step(params, policy.init_carry(), gray, iv, jax.random.PRNGKey(0))[1].rates
    type_id = jnp.asarray(neuron_types.map({t: i for i, t in enumerate(types)}), jnp.int32)
    per_type = jnp.maximum(jnp.bincount(type_id, length=len(types)), 1)

    def quantize(change: jax.Array) -> jax.Array:
        return jnp.clip(jnp.round(change * (127 / ACTIVITY_SCALE)), -127, 127).astype(jnp.int8)

    def encode(rates: jax.Array) -> tuple[jax.Array, jax.Array]:
        change = rates - rest
        means = jax.ops.segment_sum(change, type_id, num_segments=len(types)) / per_type
        return quantize(change), quantize(means)

    return encode


class EnvSaberSim:
    """One song in the JAX env, stepped one frame at a time."""

    def __init__(self, live: LiveSaber, map_data: dict[str, Any], seed: int) -> None:
        self.live = live
        self.seed = seed
        self.notes = note_arrays(map_data["notes"], MAX_NOTES)
        self.state: SaberState | None = None
        self.obs: jax.Array | None = None
        self.carry = live.carry
        self.key = jax.random.PRNGKey(seed)
        self.iv = live.interventions({})
        self.eye = eye_controls()
        self.encoded: tuple[jax.Array, jax.Array] | None = None

    def set_interventions(self, controls: Mapping[str, Any], eye: Mapping[str, Any]) -> None:
        """Replace this sim's brain controls and eye controls; they last through resets.
        Raises ValueError on bad controls, or on any change to a sim with no brain."""
        if self.live.n_neurons == 0 and (controls or any(eye.values())):
            raise ValueError("the scripted expert has no brain or eyes to change")
        self.iv = self.live.interventions(controls)
        self.eye = eye_controls(**eye)
        if self.state is not None:
            self.state = dataclasses.replace(self.state, eye=self.eye)

    def reset(self, song_t: float) -> RenderState:
        speed = jnp.float32(self.live.env.cfg.note_speed_mps)
        self.obs, self.state = self.live.reset_fn(
            self.notes, speed, jnp.float32(song_t), jnp.float32(OPEN_END_SEC), self.eye
        )
        self.carry = self.live.carry
        self.key = jax.random.PRNGKey(self.seed)
        return self.render_state()

    def step(self) -> StepOutput:
        assert self.state is not None and self.obs is not None, "reset before step"
        t = round(float(self.live.env.time(self.state)), 6)
        self.state, self.obs, self.carry, self.key, done, events, self.encoded = self.live.step_fn(
            self.state, self.obs, self.carry, self.iv, self.key
        )
        codes = np.asarray(events)
        judged = [NoteEvent(t, EVENT_KINDS[int(codes[i])], int(i)) for i in np.flatnonzero(codes)]
        return StepOutput(self.render_state(), judged, bool(done))

    def activity(self) -> tuple[bytes, bytes] | None:
        if self.live.n_neurons == 0 or self.encoded is None:
            return None
        rates, means = self.encoded
        return np.asarray(rates).tobytes(), np.asarray(means).tobytes()

    def render_state(self) -> RenderState:
        assert self.state is not None
        return RenderState.model_validate(self.live.env.render_state(self.state))
