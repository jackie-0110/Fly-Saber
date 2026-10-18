"""Map format v4 parser (spec 8.2).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from collections.abc import Iterator
from typing import Any

from fly_saber.maps.chart import Chart, ChartNote, ChartWall, UnsupportedMapError


def placed(dat: dict[str, Any], objects: str, data: str) -> Iterator[tuple[float, dict[str, Any]]]:
    """(beat, shared data) for each object; a v4 object indexes its `*Data` entry with `i`."""
    shared = dat.get(data, [])
    for obj in dat.get(objects, []):
        if obj.get("r", 0):
            raise UnsupportedMapError("rotating v4 chart")
        index = obj.get("i", 0)
        if not 0 <= index < len(shared):
            raise UnsupportedMapError(f"{objects} index {index} has no {data} entry")
        yield obj.get("b", 0), shared[index]


def parse(dat: dict[str, Any]) -> Chart:
    """Parse a v4 difficulty file into a beat-time chart.

    As in v3, writers omit any field whose value is 0. Tempo lives in the level's audio
    data file, not here.
    """
    notes = [
        ChartNote(b, m.get("x", 0), m.get("y", 0), m.get("c", 0), m.get("d", 0))
        for b, m in placed(dat, "colorNotes", "colorNotesData")
    ]
    notes += [
        ChartNote(b, m.get("x", 0), m.get("y", 0), None, None)
        for b, m in placed(dat, "bombNotes", "bombNotesData")
    ]
    walls = [
        ChartWall(b, m.get("d", 0), m.get("x", 0), m.get("y", 0), m.get("w", 0), m.get("h", 0))
        for b, m in placed(dat, "obstacles", "obstaclesData")
    ]
    return Chart(tuple(notes), tuple(walls), ())
