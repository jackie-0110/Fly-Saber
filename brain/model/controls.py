"""Turn UI brain controls into an Interventions pytree.

Owner: Person 1. The UI's choices arrive as the `interventions` object of the
`set_interventions` WebSocket message (docs/PROTOCOL.md section 4); the server
calls `from_controls` and passes the result to every brain step.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import yaml

from brain.model.interventions import Interventions, lesion, no_interventions


@dataclass(frozen=True)
class Section:
    group: str
    label: str
    description: str


@dataclass(frozen=True)
class Mood:
    name: str
    label: str
    group: str
    current: float
    low_label: str
    high_label: str
    description: str


@dataclass(frozen=True)
class ControlsConfig:
    sections: list[Section]
    moods: list[Mood]
    max_stimulation: float
    max_noise: float


def load_controls(path: str | Path = "configs/controls.yaml") -> ControlsConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ControlsConfig(
        sections=[Section(**s) for s in raw["sections"]],
        moods=[Mood(**m) for m in raw["moods"]],
        max_stimulation=float(raw["max_stimulation"]),
        max_noise=float(raw["max_noise"]),
    )


def _clamp(x: Any, lo: float, hi: float, what: str) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} must be a number, got {x!r}") from err
    return min(max(value, lo), hi)


def from_controls(
    controls: Mapping[str, Any],
    groups: Mapping[str, list[int]],
    cfg: ControlsConfig,
    n_neurons: int,
) -> Interventions:
    """Build interventions from UI state. Raises ValueError on unknown names or bad values.

    `controls` (every key optional):
        lesion: [group, ...]               silence these groups
        stimulate: {group: current, ...}   opto current, clamped to 0..max_stimulation
        mood: {mood_name: level, ...}      level clamped to -1..1, 0 is neutral
        noise: std                         clamped to 0..max_noise
    Lesion wins over stimulation and mood for neurons in both.
    """
    unknown_keys = set(controls) - {"lesion", "stimulate", "mood", "noise"}
    if unknown_keys:
        raise ValueError(f"unknown control keys {sorted(unknown_keys)}")

    def indices(group: str) -> jnp.ndarray:
        if group not in groups:
            raise ValueError(f"unknown group {group!r}")
        return jnp.asarray(groups[group], dtype=jnp.int32)

    moods = {m.name: m for m in cfg.moods}
    iv = no_interventions(n_neurons)
    opto = iv.opto
    for group, current in dict(controls.get("stimulate", {})).items():
        amount = _clamp(current, 0.0, cfg.max_stimulation, f"stimulate[{group}]")
        opto = opto.at[indices(group)].add(amount)
    for name, level in dict(controls.get("mood", {})).items():
        if name not in moods:
            raise ValueError(f"unknown mood {name!r}")
        amount = _clamp(level, -1.0, 1.0, f"mood[{name}]") * moods[name].current
        opto = opto.at[indices(moods[name].group)].add(amount)
    noise = _clamp(controls.get("noise", 0.0), 0.0, cfg.max_noise, "noise")
    iv = iv._replace(opto=opto, noise_std=jnp.asarray(noise))
    for group in list(controls.get("lesion", [])):
        iv = lesion(iv, indices(group))
        iv = iv._replace(opto=iv.opto.at[indices(group)].set(0.0))
    return iv
