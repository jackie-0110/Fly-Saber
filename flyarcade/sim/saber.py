"""Fly Saber env, pure-functional JAX (spec 7.2).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from flyarcade.sim.eye import Patches

N_LANES = 4
N_LAYERS = 3

# Indexed by sim direction (flyarcade.maps.adapt): 0 up, 1 down, 2 left, 3 right, 4 any.
# Unit vector toward the side of the block where each cut starts; that half is drawn bright.
CUT_START_SIDE = jnp.array([[0.0, -1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])


@dataclass(frozen=True)
class SaberConfig:
    lane_spacing_m: float
    layer_spacing_m: float
    block_size_m: float
    eye_distance_m: float
    bright_luminance: float
    dim_luminance: float
    bomb_luminance: float


def block_patches(
    lane: jax.Array,
    layer: jax.Array,
    distance_m: jax.Array,
    direction: jax.Array,
    is_bomb: jax.Array,
    cfg: SaberConfig,
) -> Patches:
    """Two patches per note (bright and dim halves), (2 * notes,) rows.

    `distance_m` is how far in front of the hit plane each note is. Bombs are
    one dark full-size patch; their second patch has zero size.
    """
    half = cfg.block_size_m / 2.0
    center = jnp.stack(
        [
            (lane - (N_LANES - 1) / 2.0) * cfg.lane_spacing_m,
            (layer - (N_LAYERS - 1) / 2.0) * cfg.layer_spacing_m,
            distance_m + cfg.eye_distance_m,
        ],
        axis=-1,
    )
    side = jnp.where(is_bomb[:, None], 0.0, CUT_START_SIDE[direction])
    offset = jnp.concatenate([side * half / 2.0, jnp.zeros_like(lane)[:, None]], axis=-1)
    bright_half_size = half - jnp.abs(side) * half / 2.0
    dim_half_size = bright_half_size * jnp.abs(side).sum(axis=-1, keepdims=True)
    bright = jnp.where(is_bomb, cfg.bomb_luminance, cfg.bright_luminance)
    return Patches(
        center=jnp.concatenate([center + offset, center - offset]),
        half_size=jnp.concatenate([bright_half_size, dim_half_size]),
        luminance=jnp.concatenate([bright, jnp.full_like(bright, cfg.dim_luminance)]),
    )
