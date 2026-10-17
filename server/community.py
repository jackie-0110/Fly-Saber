"""Community BeatSaver maps, fetched and normalized on demand so a brain can play them.

Owner: Person 1. Only the .dat charts are downloaded: the browser plays the audio it
fetched from BeatSaver itself, so the server never stores or serves a community song.
Maps land in their own directory, apart from the allowlisted and training maps.
"""

import json
import re
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from fly_saber.maps.beatsaver_client import download_level, make_client, map_metadata
from fly_saber.maps.normalize import normalize_level

KEY_PATTERN = re.compile(r"[a-f0-9]{1,8}")
SPLIT = "community"


class MapImportError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class CommunityMaps:
    """Normalized community charts under `cache_dir`, keyed like every other map id."""

    def __init__(self, cache_dir: Path, client: Callable[[], httpx.Client] = make_client) -> None:
        self.cache_dir = cache_dir
        self.client = client
        self.lock = threading.Lock()  # one import at a time: BeatSaver rate-limits, disk races

    def map_path(self, map_id: str) -> Path:
        return self.cache_dir / "normalized" / f"{map_id}.json"

    def fetch(self, key: str) -> list[str]:
        """Download and normalize every Standard difficulty of BeatSaver map `key` once;
        return their map ids. Raises MapImportError with an HTTP status."""
        key = key.lower()
        if not KEY_PATTERN.fullmatch(key):
            raise MapImportError(400, f"invalid BeatSaver key {key!r}")
        with self.lock:
            index = self.cache_dir / "index" / f"{key}.json"
            if index.exists():
                ids: list[str] = json.loads(index.read_text())
                return ids
            maps = self._normalize(key)
            (self.cache_dir / "normalized").mkdir(parents=True, exist_ok=True)
            for normalized in maps:
                self.map_path(normalized["map_id"]).write_text(json.dumps(normalized))
            ids = [m["map_id"] for m in maps]
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps(ids))
            return ids

    def _normalize(self, key: str) -> list[dict[str, Any]]:
        raw_dir = self.cache_dir / "raw"
        try:
            with self.client() as client:
                meta = map_metadata(client, key, raw_dir)
                level_dir = download_level(client, meta, raw_dir, charts_only=True)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            raise MapImportError(
                404 if status == 404 else 502, f"BeatSaver answered {status} for map {key}"
            ) from error
        except httpx.HTTPError as error:
            raise MapImportError(502, f"could not reach BeatSaver: {error}") from error
        except (zipfile.BadZipFile, KeyError, IndexError) as error:
            raise MapImportError(502, f"BeatSaver map {key} is malformed") from error
        try:
            maps = normalize_level(
                level_dir,
                key=key,
                split=SPLIT,
                duration_sec=float(meta["metadata"]["duration"]),
                allowed=None,
            )
        except (ValueError, KeyError, StopIteration, TypeError) as error:
            # UnsupportedMapError is a ValueError: legacy offsets, rotating charts, unknown formats.
            raise MapImportError(422, f"the fly cannot play this map yet: {error}") from error
        if not maps:
            raise MapImportError(422, "the map has no Standard difficulty for the fly to play")
        return maps
