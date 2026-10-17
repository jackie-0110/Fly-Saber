"""Hexagonal ommatidia lattice and analytic rendering (spec 7.1).

Owner: Person 2. See FLYSABER_HANDOFF.md for the spec.

Coordinates: the eye is at the origin, x right, y up, z forward. Azimuth is
positive to the right, elevation positive up.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True)
class EyeConfig:
    ommatidium_spacing_deg: float
    acceptance_fwhm_deg: float
    lateral_extent_deg: float
    binocular_overlap_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    background_luminance: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Patches:
    """Flat rectangles facing the eye; padded rows have zero half size."""

    center: jax.Array
    """(P, 3) meters."""
    half_size: jax.Array
    """(P, 2) meters, (horizontal, vertical)."""
    luminance: jax.Array
    """(P,) 0 to 1."""


def lattice(cfg: EyeConfig) -> tuple[np.ndarray, int]:
    """Ommatidia directions as (N, 2) radians (azimuth, elevation), left eye first.

    Returns the array and the left eye's count. The right eye mirrors the left.
    """
    spacing = cfg.ommatidium_spacing_deg
    row_step = spacing * math.sqrt(3.0) / 2.0
    n_rows = int((cfg.elevation_max_deg - cfg.elevation_min_deg) / row_step) + 1
    left = []
    for row in range(n_rows):
        elevation = cfg.elevation_min_deg + row * row_step
        azimuth = -cfg.lateral_extent_deg + (spacing / 2.0 if row % 2 else 0.0)
        while azimuth <= cfg.binocular_overlap_deg:
            left.append((azimuth, elevation))
            azimuth += spacing
    left_deg = np.array(left)
    right_deg = left_deg * np.array([-1.0, 1.0])
    return np.radians(np.concatenate([left_deg, right_deg])), len(left_deg)


def render(directions: jax.Array, patches: Patches, cfg: EyeConfig) -> jax.Array:
    """Luminance per ommatidium, (N,), from patches on a uniform background.

    Each patch's coverage is its angular box convolved with the Gaussian
    acceptance (separable erf), using a small-angle tangent plane per patch.
    """
    x, y, z = patches.center[:, 0], patches.center[:, 1], patches.center[:, 2]
    distance = jnp.sqrt(x**2 + y**2 + z**2)
    patch_az = jnp.arctan2(x, z)
    patch_el = jnp.arctan2(y, jnp.sqrt(x**2 + z**2))
    half_angle = jnp.arctan2(patches.half_size, distance[:, None])

    d_az = (directions[:, None, 0] - patch_az[None, :]) * jnp.cos(patch_el)[None, :]
    d_el = directions[:, None, 1] - patch_el[None, :]
    scale = math.sqrt(2.0) * math.radians(cfg.acceptance_fwhm_deg) / FWHM_PER_SIGMA

    def box(offset: jax.Array, half: jax.Array) -> jax.Array:
        return 0.5 * (erf((offset + half) / scale) - erf((offset - half) / scale))

    coverage = box(d_az, half_angle[None, :, 0]) * box(d_el, half_angle[None, :, 1])
    contrast = patches.luminance - cfg.background_luminance
    return jnp.clip(cfg.background_luminance + coverage @ contrast, 0.0, 1.0)
