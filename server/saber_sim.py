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
from brain.model.controls import from_controls, load_controls
from brain.model.interventions import Interventions
from fly_saber.config import load_config
from fly_saber.env import Event, SaberConfig, SaberEnv, SaberState
from fly_saber.experts import saber_expert
from fly_saber.eye import EyeConfig, eye_controls
from fly_saber.maps.adapt import note_arrays
from server.protocol import RenderState
from server.session import NoteEvent, StepOutput
from training.bc import action_heads, greedy_action
from training.play_eval import load_trained

Act = Callable[[SaberState, jax.Array, Any, Any, jax.Array], tuple[jax.Array, Any]]
"""(state, obs, carry, interventions, key) -> ((n_sabers, 2) action, carry) for one unbatched
env. The carry is the player's memory between frames: a brain's membrane potentials."""

EVENT_KINDS: dict[int, Literal["hit", "miss", "bad_cut", "bomb"]] = {
    Event.HIT: "hit",
    Event.MISS: "miss",
    Event.BAD_CUT: "bad_cut",
    Event.BOMB: "bomb",
}
# Past any song: the episode ends on its own after the last note.
OPEN_END_SEC = 1.0e6


class LiveSaber:
    """Sim factory for one player; the env step is compiled once per padded map size."""

    def __init__(
        self,
        env: SaberEnv,
        act: Act,
        carry: Any,
        interventions: Callable[[Mapping[str, Any]], Any] = lambda controls: (),
        n_neurons: int = 0,
        neuron_cell_types: Sequence[str] = (),
    ) -> None:
        """`interventions` turns set_interventions brain controls (PROTOCOL.md section 4)
        into what `act` takes; it raises ValueError on bad controls."""
        self.env = env
        self.carry = carry
        self.interventions = interventions
        self.n_neurons = n_neurons
        self.cell_types = list(neuron_cell_types)
        self.reset_fn = jax.jit(env.reset)

        def step(
            state: SaberState, obs: jax.Array, carry: Any, iv: Any, key: jax.Array
        ) -> tuple[SaberState, jax.Array, Any, jax.Array, jax.Array, jax.Array]:
            key, k_act = jax.random.split(key)
            action, carry = act(state, obs, carry, iv, k_act)
            obs, state, _, done, events = env.step(state, action)
            return state, obs, carry, key, done, events

        self.step_fn = jax.jit(step)

    def __call__(self, map_data: dict[str, Any], seed: int) -> "EnvSaberSim":
        return EnvSaberSim(self, map_data, seed)


def expert_sims() -> LiveSaber:
    """The real env played by the scripted expert."""
    env = SaberEnv.create(load_config(SaberConfig, "saber"), load_config(EyeConfig, "eye"))
    return LiveSaber(env, lambda state, obs, carry, iv, key: (saber_expert(env, state), carry), ())


def brain_sims(checkpoint: Path, subgraph_dir: Path) -> LiveSaber:
    """The env played greedily by a trained brain, seeing through its eye columns.

    `subgraph_dir` must hold the exact subgraph it was trained on (a bundle's subgraph/).
    """
    trained = load_trained(checkpoint, subgraph_dir)
    policy, params = trained.policy, trained.params
    heads = action_heads(trained.env.cfg.n_sabers)

    controls = load_controls()

    def act(
        state: SaberState, obs: jax.Array, v: jax.Array, iv: Interventions, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        v, logits = dataclasses.replace(policy, interventions=iv).apply(params, v, obs, key)
        return greedy_action(logits, heads), v

    def interventions(ui: Mapping[str, Any]) -> Interventions:
        return from_controls(ui, trained.groups, controls, trained.n_neurons)

    neurons = pd.read_parquet(subgraph_dir / "neurons.parquet", columns=["type"])
    return LiveSaber(
        trained.env,
        act,
        policy.init_carry(),
        interventions,
        trained.n_neurons,
        cell_types(neurons),
    )


class EnvSaberSim:
    """One song in the JAX env, stepped one frame at a time."""

    def __init__(self, live: LiveSaber, map_data: dict[str, Any], seed: int) -> None:
        self.live = live
        self.seed = seed
        notes = map_data["notes"]
        # Powers of two bound how many map sizes the step function compiles for.
        padded = max(live.env.cfg.max_visible_notes, 1 << max(len(notes) - 1, 0).bit_length())
        self.notes = note_arrays(notes, padded)
        self.state: SaberState | None = None
        self.obs: jax.Array | None = None
        self.carry = live.carry
        self.key = jax.random.PRNGKey(seed)
        self.iv = live.interventions({})
        self.eye = eye_controls()

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
        self.state, self.obs, self.carry, self.key, done, events = self.live.step_fn(
            self.state, self.obs, self.carry, self.iv, self.key
        )
        codes = np.asarray(events)
        judged = [NoteEvent(t, EVENT_KINDS[int(codes[i])], int(i)) for i in np.flatnonzero(codes)]
        return StepOutput(self.render_state(), judged, bool(done))

    def render_state(self) -> RenderState:
        assert self.state is not None
        return RenderState.model_validate(self.live.env.render_state(self.state))
