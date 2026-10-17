"""Fly eye directions (hex lattice or connectome columns) and analytic rendering (spec 7.1).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.

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
    hex1_axis_deg: float
    hex2_axis_deg: float


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


def column_directions(
    side: np.ndarray, hex1: np.ndarray, hex2: np.ndarray, cfg: EyeConfig
) -> np.ndarray:
    """Viewing directions of connectome eye columns as (n_columns, 2) radians, in row order.

    One hex step is `ommatidium_spacing_deg` along `hex1_axis_deg` or `hex2_axis_deg`,
    measured in the left eye from anterior (0) toward dorsal (90). Each eye is
    centered between its lateral edge and the binocular overlap; the right eye
    mirrors the left.
    """
    side = np.asarray(side)
    if not np.isin(side, ["L", "R"]).all():
        raise ValueError("column side must be L or R")
    axes = np.radians([cfg.hex1_axis_deg, cfg.hex2_axis_deg])
    basis = cfg.ommatidium_spacing_deg * np.stack([np.cos(axes), np.sin(axes)])
    hexes = np.stack([hex1, hex2], axis=-1).astype(np.float64)
    center_azimuth = (cfg.binocular_overlap_deg - cfg.lateral_extent_deg) / 2.0
    degrees = np.zeros((len(side), 2))
    for eye, mirror in (("L", 1.0), ("R", -1.0)):
        rows = side == eye
        offsets = hexes[rows] @ basis.T
        anterior, dorsal = (offsets - offsets.mean(axis=0)).T
        degrees[rows] = np.stack([mirror * (center_azimuth + anterior), dorsal], axis=-1)
    return np.radians(degrees)


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
