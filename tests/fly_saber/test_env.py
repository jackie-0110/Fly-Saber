"""Fly Saber env judging, expert, and batching."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fly_saber.config import load_config
from fly_saber.env import IDLE_CELL, NO_CUT, Event, SaberConfig, SaberEnv, SaberState
from fly_saber.experts import saber_expert
from fly_saber.eye import EyeConfig
from fly_saber.maps.adapt import NoteArrays, note_arrays

EYE = load_config(EyeConfig, "eye")
SPEED = jnp.float32(8.0)


def make_env(**overrides: Any) -> SaberEnv:
    return SaberEnv.create(load_config(SaberConfig, "saber", **overrides), EYE)


def note(time_sec: float, color: int | None, cut: int | None, lane: int = 1) -> dict[str, Any]:
    return {
        "time_sec": time_sec,
        "lane": lane,
        "layer": 1,
        "color": color,
        "cut_direction": cut,
        "is_bomb": color is None,
    }


# A red up-cut, a blue down-cut, a bomb in lane 3, and a red any-direction note.
NOTES = note_arrays(
    [note(1.0, 0, 0), note(1.5, 1, 1, lane=2), note(2.0, None, None, lane=3), note(2.5, 0, 8)],
    max_notes=32,
)


def play(
    env: SaberEnv, policy: Any, notes: NoteArrays = NOTES
) -> tuple[SaberState, float, list[int]]:
    _, state = env.reset(notes, SPEED, jnp.float32(0.0), jnp.float32(1e5))
    step = jax.jit(env.step)
    total, events, done = 0.0, [], False
    while not done:
        _, state, reward, done, step_events = step(state, policy(env, state))
        total += float(reward)
        events += [int(e) for e in step_events if e]
    return state, total, events


def idle(env: SaberEnv, state: SaberState) -> jax.Array:
    return jnp.tile(jnp.array([IDLE_CELL, NO_CUT]), (env.cfg.n_sabers, 1))


def test_expert_hits_every_note_and_avoids_bombs() -> None:
    env = make_env()
    state, total, events = play(env, saber_expert)
    assert events == [Event.HIT] * 3
    assert total == 3 * env.cfg.reward_hit
    assert env.render_state(state)["score"]["score"] == 100


def test_single_saber_expert_hits_both_colors() -> None:
    state, _, events = play(make_env(two_sabers=False), saber_expert)
    assert events == [Event.HIT] * 3


def test_idle_policy_misses_every_note() -> None:
    env = make_env()
    state, total, events = play(env, idle)
    assert events == [Event.MISS] * 3
    assert total == 3 * env.cfg.reward_miss


def flipped_expert(env: SaberEnv, state: SaberState) -> jax.Array:
    """Expert that cuts every up-note downward (sim directions 0 up, 1 down)."""
    action = saber_expert(env, state)
    return action.at[:, 1].set(jnp.where(action[:, 1] == 0, 1, action[:, 1]))


def test_wrong_direction_is_a_bad_cut_unless_direction_is_unscored() -> None:
    _, _, events = play(make_env(), flipped_expert)
    assert events[0] == Event.BAD_CUT
    _, _, events = play(make_env(score_direction=False), flipped_expert)
    assert events == [Event.HIT] * 3


def test_cutting_a_bomb_is_penalized() -> None:
    env = make_env()

    def bomb_cutter(env: SaberEnv, state: SaberState) -> jax.Array:
        return jnp.array([[1 * 4 + 3, 0], [IDLE_CELL, NO_CUT]])

    _, _, events = play(env, bomb_cutter)
    assert Event.BOMB in events


def test_reset_and_step_batch_under_vmap() -> None:
    env = make_env()
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), NOTES)
    starts = jnp.array([0.0, 1.2], jnp.float32)
    obs, state = jax.vmap(env.reset, in_axes=(0, None, 0, None))(
        batch, SPEED, starts, jnp.float32(1e5)
    )
    action = jax.vmap(saber_expert, in_axes=(None, 0))(env, state)
    obs, state, reward, done, events = jax.jit(jax.vmap(env.step))(state, action)
    assert obs.shape == (2, 2, len(env.directions))
    np.testing.assert_array_equal(state.step, [1, 1])


def test_observation_shows_approaching_note_and_hides_it_after_the_hit() -> None:
    env = make_env()
    background = EYE.background_luminance
    _, state = env.reset(
        note_arrays([note(1.0, 0, 0)], 32), SPEED, jnp.float32(0.9), jnp.float32(9)
    )
    assert np.abs(env.observe(state)[0] - background).max() > 0.1
    while int(state.hits) == 0:
        _, state, *_ = env.step(state, saber_expert(env, state))
    np.testing.assert_allclose(env.observe(state), background)


def test_notes_at_the_end_of_a_window_are_still_judged() -> None:
    env = make_env()
    notes = note_arrays([note(0.4 + 0.3 * i, 0, 0) for i in range(6)], max_notes=32)
    _, state = env.reset(notes, SPEED, jnp.float32(0.0), jnp.float32(2.0))
    step, done = jax.jit(env.step), False
    while not done:
        _, state, _, done, _ = step(state, idle(env, state))
    assert int(state.misses) == 6


def test_render_step_counts_from_song_start_whatever_the_reset_time() -> None:
    env = make_env()
    _, state = env.reset(NOTES, SPEED, jnp.float32(1.0), jnp.float32(5.0))
    assert env.render_state(state)["step"] == 30
    _, state, *_ = env.step(state, idle(env, state))
    assert env.render_state(state)["step"] == 31
