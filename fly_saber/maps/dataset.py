"""Stack normalized maps into one padded batch and sample training windows (spec 7.2, 12).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from fly_saber.maps.adapt import NoteArrays, note_arrays, thin_notes


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MapSet:
    """Every map of one split, notes padded to a common length; leading axis is the map."""

    notes: NoteArrays
    speed_mps: jax.Array
    duration_sec: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Episodes:
    """Arguments for `SaberEnv.reset`, one row per env."""

    notes: NoteArrays
    speed_mps: jax.Array
    start_sec: jax.Array
    end_sec: jax.Array


def load_map_set(
    normalized_dir: Path, split: str, min_notes: int, min_gap_sec: float = 0.0
) -> tuple[MapSet, list[str]]:
    """Load every `split` map, thinned to `min_gap_sec`; also returns their map ids.

    `min_notes` is the smallest padded length, normally the env's `max_visible_notes`.
    """
    maps = [json.loads(p.read_text()) for p in sorted(normalized_dir.glob("*.json"))]
    maps = [m for m in maps if m["split"] == split]
    if not maps:
        raise ValueError(f"no {split} maps in {normalized_dir}")
    notes = [thin_notes(m["notes"], min_gap_sec) for m in maps]
    max_notes = max(min_notes, *(len(n) for n in notes))
    arrays = [note_arrays(n, max_notes) for n in notes]
    map_set = MapSet(
        notes=jax.tree.map(lambda *x: jnp.stack(x), *arrays),
        speed_mps=jnp.array([m["difficulty"]["note_speed_mps"] for m in maps], jnp.float32),
        duration_sec=jnp.array([m["song"]["duration_sec"] for m in maps], jnp.float32),
    )
    return map_set, [m["map_id"] for m in maps]


def sample_episodes(key: jax.Array, map_set: MapSet, n: int, window_sec: float) -> Episodes:
    """`n` random (map, start time) windows of `window_sec`; jit-safe."""
    k_map, k_start = jax.random.split(key)
    index = jax.random.randint(k_map, (n,), 0, map_set.speed_mps.shape[0])
    latest_start = jnp.maximum(map_set.duration_sec[index] - window_sec, 0.0)
    start = jax.random.uniform(k_start, (n,)) * latest_start
    return Episodes(
        notes=jax.tree.map(lambda x: x[index], map_set.notes),
        speed_mps=map_set.speed_mps[index],
        start_sec=start,
        end_sec=start + window_sec,
    )
