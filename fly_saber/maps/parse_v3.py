"""Map format v3 parser (spec 8.2).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.
"""

from typing import Any

from fly_saber.maps.chart import BpmChange, Chart, ChartNote, ChartWall


def parse(dat: dict[str, Any]) -> Chart:
    """Parse a v3 difficulty file into a beat-time chart.

    v3 writers may omit any field whose value is 0, so every read defaults to 0.
    """
    notes = [
        ChartNote(n.get("b", 0), n.get("x", 0), n.get("y", 0), n.get("c", 0), n.get("d", 0))
        for n in dat.get("colorNotes", [])
    ]
    notes += [
        ChartNote(b.get("b", 0), b.get("x", 0), b.get("y", 0), None, None)
        for b in dat.get("bombNotes", [])
    ]
    walls = [
        ChartWall(
            o.get("b", 0), o.get("d", 0), o.get("x", 0), o.get("y", 0), o.get("w", 0), o.get("h", 0)
        )
        for o in dat.get("obstacles", [])
    ]
    bpm_changes = [BpmChange(e.get("b", 0), e["m"]) for e in dat.get("bpmEvents", [])]
    return Chart(tuple(notes), tuple(walls), tuple(bpm_changes))
