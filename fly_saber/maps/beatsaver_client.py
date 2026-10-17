"""Cached BeatSaver API client (spec 8.1). Never called at demo time.

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://api.beatsaver.com"
USER_AGENT = "FlySaber/0.1 (SwampHacks)"


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def map_metadata(client: httpx.Client, key: str, cache_dir: Path) -> dict[str, Any]:
    """BeatSaver map metadata, cached at `cache_dir/{key}/beatsaver.json`."""
    cache = cache_dir / key / "beatsaver.json"
    if not cache.exists():
        response = client.get(f"{API_URL}/maps/id/{key}")
        response.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(response.text)
    meta: dict[str, Any] = json.loads(cache.read_text())
    return meta


def download_level(client: httpx.Client, meta: dict[str, Any], cache_dir: Path) -> Path:
    """Download and extract the map's latest version once; return the level directory."""
    level_dir: Path = cache_dir / meta["id"]
    if not any(p.name.lower() == "info.dat" for p in level_dir.glob("*")):
        response = client.get(meta["versions"][0]["downloadURL"])
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(level_dir)
    return level_dir
