"""Lesion, opto, and mood interventions as one pytree (spec 6.4).

Owner: Person 1. Applied at inference time with no retraining; the server
builds one of these from UI state and passes it into every brain step.
Eye controls (blind, blur, noise) are applied in src/fly_saber/eye.py, before
the brain sees the input.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Interventions(NamedTuple):
    gain: Array  # (N,) output multiplier per neuron; 0 lesions it
    opto: Array  # (N,) extra current per neuron; large positive forces it on
    noise_std: Array  # () std of current noise added to every neuron each substep


def no_interventions(n_neurons: int) -> Interventions:
    """The unmodified brain."""
    return Interventions(
        gain=jnp.ones(n_neurons), opto=jnp.zeros(n_neurons), noise_std=jnp.zeros(())
    )


def lesion(iv: Interventions, indices: Array) -> Interventions:
    """Silence the neurons at `indices` (for example a group from groups.json)."""
    return iv._replace(gain=iv.gain.at[indices].set(0.0))


def stimulate(iv: Interventions, indices: Array, current: float) -> Interventions:
    """Optogenetically drive the neurons at `indices` with `current`."""
    return iv._replace(opto=iv.opto.at[indices].set(current))
