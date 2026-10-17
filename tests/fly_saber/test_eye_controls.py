"""Live eye controls: blind, blur, facet dropout, noise."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fly_saber.config import load_config
from fly_saber.env import SaberConfig, SaberEnv, SaberState, set_eye_controls
from fly_saber.eye import EyeConfig, apply_eye_controls, eye_controls
from fly_saber.maps.adapt import note_arrays

EYE = load_config(EyeConfig, "eye")
ENV = SaberEnv.create(load_config(SaberConfig, "saber"), EYE)
BACKGROUND = EYE.background_luminance


def note_state(lane: int = 1) -> SaberState:
    """One red note in `lane`, close enough that both eyes see it."""
    note = {
        "time_sec": 1.0,
        "lane": lane,
        "layer": 1,
        "color": 0,
        "cut_direction": 0,
        "is_bomb": False,
    }
    notes = note_arrays([note], 32)
    _, state = ENV.reset(notes, jnp.float32(8.0), jnp.float32(0.9), jnp.float32(9))
    return state


def observe(state: SaberState, **controls: Any) -> np.ndarray:
    return np.asarray(ENV.observe(set_eye_controls(state, **controls)))


def test_default_controls_leave_the_image_unchanged() -> None:
    image = jax.random.uniform(jax.random.PRNGKey(0), (2, len(ENV.directions)))
    out = apply_eye_controls(image, eye_controls(), ENV.left_eye, jnp.int32(3), BACKGROUND)
    np.testing.assert_array_equal(out, image)


def test_blind_left_flattens_only_left_columns() -> None:
    state = note_state()
    seen, blind = observe(state), observe(state, blind_left=1.0)
    left = np.asarray(ENV.left_eye)
    assert np.abs(seen[:, left] - BACKGROUND).max() > 0.1
    np.testing.assert_allclose(blind[:, left], BACKGROUND)
    np.testing.assert_array_equal(blind[:, ~left], seen[:, ~left])


def test_facet_dropout_drops_the_same_columns_every_frame_for_a_seed() -> None:
    image = jnp.ones((2, len(ENV.directions)))

    def dropped(seed: int, step: int) -> np.ndarray:
        controls = eye_controls(facet_dropout=0.3, seed=seed)
        out = apply_eye_controls(image, controls, ENV.left_eye, jnp.int32(step), BACKGROUND)
        return np.asarray(out[0] == BACKGROUND)

    first = dropped(1, 0)
    np.testing.assert_array_equal(dropped(1, 7), first)
    assert (dropped(2, 0) != first).any()
    assert abs(first.mean() - 0.3) < 0.08


def test_blur_widens_a_blocks_footprint() -> None:
    state = note_state()

    def footprint(obs: np.ndarray) -> int:
        return int((np.abs(obs[0] - BACKGROUND) > 0.01).sum())

    assert footprint(observe(state, blur_deg=10.0)) > footprint(observe(state))


def test_noise_has_the_requested_std_on_luminance_only() -> None:
    _, blank = ENV.reset(note_arrays([], 32), jnp.float32(8.0), jnp.float32(0.0), jnp.float32(9))
    obs = observe(blank, noise=0.05)
    assert abs(obs[0].std() - 0.05) < 0.005
    np.testing.assert_array_equal(obs[1], BACKGROUND)


def test_controls_change_between_frames_without_recompiling_and_batch() -> None:
    traces = []

    def traced_step(state: SaberState, action: jax.Array) -> Any:
        traces.append(1)
        return ENV.step(state, action)

    step = jax.jit(traced_step)
    state = note_state()
    action = jnp.array([[12, 4], [12, 4]])
    _, state, *_ = step(state, action)
    obs, state, *_ = step(set_eye_controls(state, blind_right=1.0, blur_deg=5, noise=0.1), action)
    assert len(traces) == 1

    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    blind = jax.vmap(lambda s, b: set_eye_controls(s, blind_left=b))(batch, jnp.array([0.0, 1.0]))
    obs = np.asarray(jax.vmap(ENV.observe)(blind))
    left = np.asarray(ENV.left_eye)
    assert np.abs(obs[0][:, left] - BACKGROUND).max() > 0.1
    np.testing.assert_allclose(obs[1][:, left], BACKGROUND)
