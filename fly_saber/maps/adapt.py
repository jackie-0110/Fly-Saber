"""Adapt normalized maps for the fly: directions, density, difficulty score (spec 8.3).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ANY_DIRECTION = 4
# Map cut direction (0 to 8) to sim direction: 0 up, 1 down, 2 left, 3 right, 4 any.
# Diagonals keep their vertical component; up and down cuts dominate real maps.
SIM_DIRECTION = np.array([0, 1, 2, 3, 0, 0, 1, 1, ANY_DIRECTION])
BOMB_COLOR = -1
# Far past any song, so padded time_sec stays sorted for searchsorted.
PAD_TIME_SEC = 1.0e6


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NoteArrays:
    """One map's notes padded to a fixed length so many maps batch under vmap."""

    time_sec: jax.Array
    lane: jax.Array
    layer: jax.Array
    color: jax.Array
    """0 red, 1 blue, -1 bomb."""
    direction: jax.Array
    is_bomb: jax.Array
    valid: jax.Array
    """False on padding rows."""


def thin_notes(notes: list[dict[str, Any]], min_gap_sec: float) -> list[dict[str, Any]]:
    """Drop a colored note if the same color had a kept note less than `min_gap_sec` before."""
    last_time: dict[int, float] = {}
    kept = []
    for note in notes:
        color = note["color"]
        if color is not None and note["time_sec"] - last_time.get(color, -np.inf) < min_gap_sec:
            continue
        if color is not None:
            last_time[color] = note["time_sec"]
        kept.append(note)
    return kept


def note_arrays(notes: list[dict[str, Any]], max_notes: int) -> NoteArrays:
    """Pad normalized notes into fixed-size arrays; walls are not part of the sim."""
    if len(notes) > max_notes:
        raise ValueError(f"{len(notes)} notes exceed max_notes={max_notes}")
    pad = max_notes - len(notes)

    def column(values: list[Any], dtype: type, fill: Any = 0) -> jax.Array:
        return jnp.asarray(np.pad(np.array(values, dtype=dtype), (0, pad), constant_values=fill))

    return NoteArrays(
        time_sec=column([n["time_sec"] for n in notes], np.float32, PAD_TIME_SEC),
        lane=column([n["lane"] for n in notes], np.int32),
        layer=column([n["layer"] for n in notes], np.int32),
        color=column([BOMB_COLOR if n["is_bomb"] else n["color"] for n in notes], np.int32),
        direction=column(
            [0 if n["is_bomb"] else SIM_DIRECTION[n["cut_direction"]] for n in notes], np.int32
        ),
        is_bomb=column([n["is_bomb"] for n in notes], np.bool_),
        valid=column([True] * len(notes), np.bool_),
    )


def difficulty(notes: list[dict[str, Any]]) -> dict[str, float]:
    """Our own difficulty meter: density, plus how often each saber changes direction or cell.

    `score` is notes per second scaled up by the direction- and cell-change fractions.
    """
    colored = [n for n in notes if not n["is_bomb"]]
    if len(colored) < 2:
        return {"notes_per_sec": 0.0, "direction_changes": 0.0, "cell_changes": 0.0, "score": 0.0}
    span = colored[-1]["time_sec"] - colored[0]["time_sec"]
    pairs = [
        pair for color in (0, 1) for pair in pairwise(n for n in colored if n["color"] == color)
    ]
    direction_changes = np.mean(
        [SIM_DIRECTION[a["cut_direction"]] != SIM_DIRECTION[b["cut_direction"]] for a, b in pairs]
    )
    cell_changes = np.mean([(a["lane"], a["layer"]) != (b["lane"], b["layer"]) for a, b in pairs])
    notes_per_sec = len(colored) / span if span > 0 else 0.0
    return {
        "notes_per_sec": float(notes_per_sec),
        "direction_changes": float(direction_changes),
        "cell_changes": float(cell_changes),
        "score": float(notes_per_sec * (1.0 + direction_changes + cell_changes)),
    }
