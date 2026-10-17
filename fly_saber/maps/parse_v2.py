"""Map format v2 parser (spec 8.2).

Owner: Person 2. See FLYSABER_HANDOFF.md for the spec.
"""

from typing import Any

from fly_saber.maps.chart import Chart, ChartNote, ChartWall, UnsupportedMapError

BOMB_TYPE = 3
COLOR_TYPES = (0, 1)
# v2 wall _type to the v3 (layer, height) the game converts it to.
WALL_SHAPES = {0: (0, 5), 1: (2, 3)}


def parse(dat: dict[str, Any]) -> Chart:
    """Parse a v2 difficulty file into a beat-time chart."""
    custom = dat.get("_customData") or {}
    if dat.get("_BPMChanges") or custom.get("_BPMChanges"):
        raise UnsupportedMapError("v2 _BPMChanges have editor-specific semantics")

    notes = []
    for n in dat.get("_notes", []):
        kind = n["_type"]
        if kind == BOMB_TYPE:
            color, cut = None, None
        elif kind in COLOR_TYPES:
            color, cut = kind, n["_cutDirection"]
        else:
            continue
        notes.append(ChartNote(n["_time"], n["_lineIndex"], n["_lineLayer"], color, cut))

    walls = []
    for o in dat.get("_obstacles", []):
        shape = WALL_SHAPES.get(o["_type"])
        if shape is None:
            continue
        layer, height = shape
        walls.append(
            ChartWall(o["_time"], o["_duration"], o["_lineIndex"], layer, o["_width"], height)
        )

    return Chart(tuple(notes), tuple(walls), ())
