"""Scripted perfect players reading the true future state (spec 7.4).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.
"""

import jax
import jax.numpy as jnp

from fly_saber.env import IDLE_CELL, N_LANES, NO_CUT, SaberEnv, SaberState
from fly_saber.maps.adapt import ANY_DIRECTION


def saber_expert(env: SaberEnv, state: SaberState) -> jax.Array:
    """Each saber waits in its next note's cell and cuts on the frame nearest the note's time."""
    cfg, notes = env.cfg, state.notes
    t = env.time(state)
    pending = ~state.judged & ~notes.is_bomb
    note_cell = notes.layer * N_LANES + notes.lane

    def plan(saber: jax.Array) -> jax.Array:
        mine = pending & ((notes.color == saber) | (not cfg.two_sabers))
        has_note = mine.any()
        target = jnp.argmax(mine)
        due = has_note & (notes.time_sec[target] - t <= 0.5 / cfg.step_hz)
        direction = notes.direction[target]
        cut = jnp.where(direction == ANY_DIRECTION, 0, direction)
        return jnp.stack(
            [jnp.where(has_note, note_cell[target], IDLE_CELL), jnp.where(due, cut, NO_CUT)]
        )

    return jax.vmap(plan)(jnp.arange(cfg.n_sabers))
