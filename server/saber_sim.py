"""The real Fly Saber env behind the session's Sim interface (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from fly_saber.env import Event, SaberEnv, SaberState
from fly_saber.maps.adapt import note_arrays
from server.protocol import RenderState
from server.session import NoteEvent, StepOutput

Controller = Callable[[SaberEnv, SaberState], jax.Array]
"""Chooses the (n_sabers, 2) action for one unbatched state: the expert, or a brain."""

EVENT_KINDS: dict[int, Literal["hit", "miss", "bad_cut", "bomb"]] = {
    Event.HIT: "hit",
    Event.MISS: "miss",
    Event.BAD_CUT: "bad_cut",
    Event.BOMB: "bomb",
}
# Past any song: the episode ends on its own after the last note.
OPEN_END_SEC = 1.0e6


class LiveSaber:
    """Sim factory for sessions; the env step is compiled once per padded map size."""

    def __init__(self, env: SaberEnv, controller: Controller) -> None:
        self.env = env
        self.reset_fn = jax.jit(env.reset)
        self.step_fn = jax.jit(lambda state: env.step(state, controller(env, state)))

    def __call__(self, map_data: dict[str, Any], brain: str, seed: int) -> "EnvSaberSim":
        return EnvSaberSim(self, map_data)


class EnvSaberSim:
    """One song in the JAX env, stepped one frame at a time."""

    def __init__(self, live: LiveSaber, map_data: dict[str, Any]) -> None:
        self.live = live
        notes = map_data["notes"]
        # Powers of two bound how many map sizes the step function compiles for.
        padded = max(live.env.cfg.max_visible_notes, 1 << max(len(notes) - 1, 0).bit_length())
        self.notes = note_arrays(notes, padded)
        self.state: SaberState | None = None

    def reset(self, song_t: float) -> RenderState:
        speed = jnp.float32(self.live.env.cfg.note_speed_mps)
        _, self.state = self.live.reset_fn(
            self.notes, speed, jnp.float32(song_t), jnp.float32(OPEN_END_SEC)
        )
        return self.render_state()

    def step(self) -> StepOutput:
        assert self.state is not None, "reset before step"
        t = round(float(self.live.env.time(self.state)), 6)
        _, self.state, _, done, events = self.live.step_fn(self.state)
        codes = np.asarray(events)
        judged = [NoteEvent(t, EVENT_KINDS[int(codes[i])], int(i)) for i in np.flatnonzero(codes)]
        return StepOutput(self.render_state(), judged, bool(done))

    def render_state(self) -> RenderState:
        assert self.state is not None
        return RenderState.model_validate(self.live.env.render_state(self.state))
