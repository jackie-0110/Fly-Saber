"""Hexagonal ommatidia lattice and analytic rendering (spec 7.1).

Owner: Person 2. See docs/project/FLY_SABER_HANDOFF.md for the spec.

Coordinates: the eye is at the origin, x right, y up, z forward. Azimuth is
positive to the right, elevation positive up.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

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


def render(
    directions: jax.Array, patches: Patches, cfg: EyeConfig, blur_deg: jax.Array | float = 0.0
) -> jax.Array:
    """Luminance per ommatidium, (N,), from patches on a uniform background.

    `blur_deg` widens the acceptance FWHM beyond `cfg.acceptance_fwhm_deg`.

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
    scale = math.sqrt(2.0) * jnp.radians(cfg.acceptance_fwhm_deg + blur_deg) / FWHM_PER_SIGMA

    def box(offset: jax.Array, half: jax.Array) -> jax.Array:
        return 0.5 * (erf((offset + half) / scale) - erf((offset - half) / scale))

    coverage = box(d_az, half_angle[None, :, 0]) * box(d_el, half_angle[None, :, 1])
    contrast = patches.luminance - cfg.background_luminance
    return jnp.clip(cfg.background_luminance + coverage @ contrast, 0.0, 1.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EyeControls:
    """Live eye perturbations (spec 7.1), all scalar arrays; `eye_controls()` sees normally."""

    blind_left: jax.Array
    """Fraction of contrast the left eye loses, 0 to 1."""
    blind_right: jax.Array
    blur_deg: jax.Array
    """Added to the acceptance FWHM."""
    facet_dropout: jax.Array
    """Fraction of columns that read as background; the same columns for a given seed."""
    noise: jax.Array
    """Std of Gaussian noise on the luminance channel, fresh each frame."""
    seed: jax.Array
    """Picks the dropped columns and the noise."""


def eye_controls(controls: EyeControls | None = None, **values: Any) -> EyeControls:
    """`controls` (default: none applied) with `values` replaced.

    Values are cast to the fields' dtypes, so a jitted step never retraces when they change.
    """
    if controls is None:
        zero = jnp.zeros((), jnp.float32)
        controls = EyeControls(zero, zero, zero, zero, zero, jnp.zeros((), jnp.uint32))
    fields = {k: jnp.asarray(v, getattr(controls, k).dtype) for k, v in values.items()}
    return dataclasses.replace(controls, **fields)


def apply_eye_controls(
    image: jax.Array,
    controls: EyeControls,
    left_eye: jax.Array,
    step: jax.Array,
    background: float,
) -> jax.Array:
    """Noise, then blinding and dropout, on a (channels, N) image; `left_eye` is (N,) bool.

    Lost contrast fades to `background`, so a blinded column looks like empty sky, not
    a black object the brain would react to.
    """
    drop_key, noise_key = jax.random.split(jax.random.PRNGKey(controls.seed))
    noise = controls.noise * jax.random.normal(jax.random.fold_in(noise_key, step), left_eye.shape)
    image = image.at[0].set(jnp.clip(image[0] + noise, 0.0, 1.0))
    kept = jax.random.uniform(drop_key, left_eye.shape) >= controls.facet_dropout
    blind = jnp.where(left_eye, controls.blind_left, controls.blind_right)
    gain = kept * (1.0 - blind)
    return image * gain + background * (1.0 - gain)
