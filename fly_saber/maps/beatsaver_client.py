"""Cached BeatSaver API client (spec 8.1). Never called at demo time.

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

import hashlib
import io
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://api.beatsaver.com"
USER_AGENT = "FlySaber/0.1 (SwampHacks)"


def make_client(min_interval_sec: float = 0.0) -> httpx.Client:
    """HTTP client that spaces its requests at least `min_interval_sec` apart."""
    last_request = 0.0

    def throttle(_: httpx.Request) -> None:
        nonlocal last_request
        time.sleep(max(0.0, last_request + min_interval_sec - time.monotonic()))
        last_request = time.monotonic()

    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=60,
        follow_redirects=True,
        event_hooks={"request": [throttle]},
    )


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


def search_maps(
    client: httpx.Client, params: dict[str, Any], cache_dir: Path
) -> Iterator[dict[str, Any]]:
    """Yield map metadata from /search/text page by page until the results run out.

    Each page is cached under `cache_dir/search/`, and each map's metadata where
    `map_metadata` looks for it, so a rerun makes no requests.
    """
    query = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    page = 0
    while True:
        cache = cache_dir / "search" / query / f"{page}.json"
        if not cache.exists():
            response = client.get(f"{API_URL}/search/text/{page}", params=params)
            response.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(response.text)
        result = json.loads(cache.read_text())
        for doc in result["docs"]:
            meta_cache = cache_dir / doc["id"] / "beatsaver.json"
            if not meta_cache.exists():
                meta_cache.parent.mkdir(parents=True, exist_ok=True)
                meta_cache.write_text(json.dumps(doc))
            yield doc
        page += 1
        if not result["docs"] or page >= result["info"]["pages"]:
            return


def download_level(
    client: httpx.Client, meta: dict[str, Any], cache_dir: Path, charts_only: bool = False
) -> Path:
    """Download and extract the map's latest version once; return the level directory.

    With `charts_only`, only the .dat files are extracted: no audio or art touches disk.
    """
    level_dir: Path = cache_dir / meta["id"]
    if not any(p.name.lower() == "info.dat" for p in level_dir.glob("*")):
        response = client.get(meta["versions"][0]["downloadURL"])
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            if charts_only:
                names = [n for n in names if n.lower().endswith(".dat")]
            archive.extractall(level_dir, members=names)
    return level_dir
