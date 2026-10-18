"""Demo map allowlist: the only maps whose audio may ship (spec 8.4).

Owner: Person 2. See docs/project/FLY_SABER_HANDOFF.md for the spec.
"""

import json
from dataclasses import dataclass
from pathlib import Path

ALLOWLIST_PATH = Path("data/maps/demo_allowlist.json")


@dataclass(frozen=True)
class AllowlistEntry:
    key: str
    """BeatSaver map key."""
    title: str
    artist: str
    license: str
    song_url: str
    """Where the song's license is published, for attribution."""
    split: str

    @property
    def attribution(self) -> str:
        return f'"{self.title}" by {self.artist} ({self.song_url}), licensed under {self.license}'


def load_allowlist(path: Path = ALLOWLIST_PATH) -> dict[str, AllowlistEntry]:
    """Allowlist entries keyed by BeatSaver key."""
    entries = [AllowlistEntry(**e) for e in json.loads(path.read_text())]
    return {e.key: e for e in entries}
