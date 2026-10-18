"""Normalized maps and allowlisted audio on disk (docs/reference/PROTOCOL.md sections 1 and 3).

Owner: Person 2. See docs/archive/project/FLY_SABER_HANDOFF.md for the spec.
"""

import json
import re
from pathlib import Path
from typing import Any

from fly_saber.maps.allowlist import load_allowlist
from server.community import CommunityMaps
from server.protocol import MapSummary

# Also keeps request paths from escaping the maps directory.
MAP_ID_PATTERN = re.compile(r"[a-z0-9]+_[a-z0-9]+")


def audio_key(map_id: str) -> str:
    return map_id.rsplit("_", 1)[0]


class Catalog:
    """Read-only view of `{maps_dir}/normalized`, `{maps_dir}/audio` and the demo allowlist,
    plus community maps fetched on demand into `{maps_dir}/community` (charts only)."""

    def __init__(self, maps_dir: Path, community: CommunityMaps | None = None) -> None:
        self.maps_dir = maps_dir
        self.community = community or CommunityMaps(maps_dir / "community")

    def allowlist(self) -> set[str]:
        return set(load_allowlist(self.maps_dir / "demo_allowlist.json"))

    def get_map(self, map_id: str) -> dict[str, Any] | None:
        if not MAP_ID_PATTERN.fullmatch(map_id):
            return None
        path = self.maps_dir / "normalized" / f"{map_id}.json"
        if not path.exists():
            path = self.community.map_path(map_id)
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def list_maps(self) -> list[MapSummary]:
        allowed = self.allowlist()
        summaries = []
        for path in sorted((self.maps_dir / "normalized").glob("*.json")):
            if audio_key(path.stem) not in allowed:
                continue
            data = json.loads(path.read_text())
            summaries.append(
                MapSummary.model_validate(
                    {
                        "map_id": data["map_id"],
                        "split": data["split"],
                        "song": {
                            k: data["song"][k] for k in ("title", "artist", "mapper", "license")
                        },
                        "difficulty": {"name": data["difficulty"]["name"]},
                        "note_count": len(data["notes"]),
                    }
                )
            )
        return summaries

    def audio_path(self, map_id: str) -> Path | None:
        if not MAP_ID_PATTERN.fullmatch(map_id) or audio_key(map_id) not in self.allowlist():
            return None
        path = self.maps_dir / "audio" / f"{audio_key(map_id)}.ogg"
        return path if path.exists() else None
