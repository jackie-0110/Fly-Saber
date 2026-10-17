"""Fly eye lattice and analytic rendering."""

import jax.numpy as jnp
import numpy as np

from fly_saber.config import load_config
from fly_saber.env import SaberConfig, block_patches
from fly_saber.eye import EyeConfig, lattice, render

EYE = load_config(EyeConfig, "eye")
SABER = load_config(SaberConfig, "saber")
DIRECTIONS, N_LEFT = lattice(EYE)


def view(direction: int, *, lane: int = 1, distance: float = 1.0, bomb: bool = False) -> np.ndarray:
    patches = block_patches(
        jnp.array([lane]),
        jnp.array([1]),
        jnp.array([distance]),
        jnp.array([direction]),
        jnp.array([bomb]),
        SABER,
    )
    return np.asarray(render(jnp.asarray(DIRECTIONS), patches, EYE))


def test_right_eye_mirrors_left() -> None:
    left, right = DIRECTIONS[:N_LEFT], DIRECTIONS[N_LEFT:]
    np.testing.assert_allclose(right, left * np.array([-1.0, 1.0]))


def test_bright_half_is_where_the_cut_starts() -> None:
    elevation = DIRECTIONS[:, 1]
    up, down = view(0), view(1)
    assert elevation[up.argmax()] < 0 < elevation[down.argmax()]


def test_bomb_is_darker_than_background_and_note_is_brighter() -> None:
    background = EYE.background_luminance
    assert view(0, bomb=True).min() < background < view(4).max()


def test_far_block_lights_fewer_ommatidia_than_near_block() -> None:
    def lit(luminance: np.ndarray) -> int:
        return int((np.abs(luminance - EYE.background_luminance) > 0.02).sum())

    assert lit(view(0, distance=8.0)) < lit(view(0, distance=1.0))
