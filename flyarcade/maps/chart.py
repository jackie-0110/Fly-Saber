"""Version-independent chart in beat time, produced by the v2 and v3 parsers.

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.
"""

from dataclasses import dataclass


class UnsupportedMapError(ValueError):
    """The map uses a feature or format version we do not parse."""


@dataclass(frozen=True)
class ChartNote:
    beat: float
    lane: int
    layer: int
    color: int | None
    """0 red, 1 blue, None for bombs."""
    cut_direction: int | None
    """0 to 8 in Beat Saber numbering, None for bombs."""


@dataclass(frozen=True)
class ChartWall:
    beat: float
    duration_beats: float
    lane: int
    layer: int
    width: int
    height: int


@dataclass(frozen=True)
class BpmChange:
    beat: float
    bpm: float


@dataclass(frozen=True)
class Chart:
    notes: tuple[ChartNote, ...]
    walls: tuple[ChartWall, ...]
    bpm_changes: tuple[BpmChange, ...]
