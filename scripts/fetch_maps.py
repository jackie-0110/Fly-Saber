"""Download allowlisted demo maps, normalize them, and stage their audio for the server.

Owner: Person 2. See FLYSABER_HANDOFF.md for the spec.
"""

import json
import shutil
from pathlib import Path

import httpx

from fly_saber.maps.allowlist import AllowlistEntry, load_allowlist
from fly_saber.maps.beatsaver_client import download_level, make_client, map_metadata
from fly_saber.maps.normalize import load_info, normalize_level

RAW_DIR = Path("data/maps/raw")
NORMALIZED_DIR = Path("data/maps/normalized")
AUDIO_DIR = Path("data/maps/audio")


def stage_level(client: httpx.Client, entry: AllowlistEntry) -> list[str]:
    """Write every difficulty of one allowlisted level; return the map ids written."""
    meta = map_metadata(client, entry.key, RAW_DIR)
    level_dir = download_level(client, meta, RAW_DIR)
    maps = normalize_level(
        level_dir,
        key=entry.key,
        split=entry.split,
        duration_sec=float(meta["metadata"]["duration"]),
        allowed=entry,
    )
    for normalized in maps:
        (NORMALIZED_DIR / f"{normalized['map_id']}.json").write_text(json.dumps(normalized))
    # The .egg files BeatSaver ships are plain Ogg Vorbis.
    shutil.copyfile(
        level_dir / load_info(level_dir)["_songFilename"], AUDIO_DIR / f"{entry.key}.ogg"
    )
    return [m["map_id"] for m in maps]


def main() -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with make_client() as client:
        for entry in load_allowlist().values():
            print(entry.key, entry.title, entry.split, stage_level(client, entry))


if __name__ == "__main__":
    main()
