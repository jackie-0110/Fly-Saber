"""Scrape a charts-only training set from BeatSaver (spec 8.1, 8.4).

Owner: Person 2. See docs/archive/project/FLY_SABER_HANDOFF.md for the spec.

Training maps never keep audio: only the .dat files are extracted and every map
is normalized without an allowlist entry, so its audio_url is null.
"""

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from fly_saber.maps.adapt import difficulty
from fly_saber.maps.allowlist import AllowlistEntry
from fly_saber.maps.beatsaver_client import download_level, map_metadata, search_maps
from fly_saber.maps.chart import MAP_ERRORS
from fly_saber.maps.normalize import normalize_level

MOD_FLAGS = ("chroma", "ne", "me", "vivify")
CURRICULUM_DIFFICULTIES = {"Easy", "Normal", "Hard"}
# What a malformed upload can raise from download through normalization.
REJECTED = (*MAP_ERRORS, zipfile.BadZipFile, httpx.HTTPStatusError)


@dataclass(frozen=True)
class TrainingMapsConfig:
    raw_dir: str
    out_dir: str
    target_songs: int
    min_rating: float
    min_upvotes: int
    page_size: int
    heldout_fraction: float
    request_interval_sec: float


def search_params(cfg: TrainingMapsConfig) -> dict[str, Any]:
    # Leaving `automapper` out returns human maps only; false would mean AI maps only.
    return {
        "minRating": cfg.min_rating,
        "minUpVotes": cfg.min_upvotes,
        "characteristics": "Standard",
        "order": "Rating",
        "pageSize": cfg.page_size,
    }


def skip_reason(meta: dict[str, Any]) -> str | None:
    """Why a search result is not worth downloading, or None to keep it."""
    if meta.get("automapper") or meta.get("declaredAi", "None") != "None":
        return "automapped"
    standard = [d for d in meta["versions"][0]["diffs"] if d["characteristic"] == "Standard"]
    if not standard:
        return "no Standard difficulty"
    if any(d.get(flag) for d in standard for flag in MOD_FLAGS):
        return "uses mods"
    if not CURRICULUM_DIFFICULTIES & {d["difficulty"] for d in standard}:
        return "no Easy, Normal, or Hard"
    return None


def song_split(key: str, heldout_fraction: float) -> str:
    """Train or heldout for every difficulty of a song, stable across runs and machines."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "heldout" if bucket < heldout_fraction else "train"


def stage_song(
    client: httpx.Client, meta: dict[str, Any], split: str, raw_dir: Path, out_dir: Path
) -> list[dict[str, Any]]:
    """Normalize one song's charts into `out_dir/normalized/`; return its index rows."""
    level_dir = download_level(client, meta, raw_dir, charts_only=True)
    maps = normalize_level(
        level_dir,
        key=meta["id"],
        split=split,
        duration_sec=float(meta["metadata"]["duration"]),
        allowed=None,
    )
    (out_dir / "normalized").mkdir(parents=True, exist_ok=True)
    rows = []
    for normalized in maps:
        (out_dir / "normalized" / f"{normalized['map_id']}.json").write_text(json.dumps(normalized))
        rows.append(
            {
                "map_id": normalized["map_id"],
                "key": meta["id"],
                "split": split,
                "difficulty": normalized["difficulty"]["name"],
                "note_count": len(normalized["notes"]),
                "score": round(difficulty(normalized["notes"])["score"], 3),
            }
        )
    return rows


def build_training_set(
    client: httpx.Client, cfg: TrainingMapsConfig, allowlist: dict[str, AllowlistEntry]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Stage the demo songs plus `target_songs` searched songs; return index rows and skips.

    Demo songs keep their allowlist split, so the unseen-map badge and training agree.
    """
    raw_dir, out_dir = Path(cfg.raw_dir), Path(cfg.out_dir)
    index: list[dict[str, Any]] = []
    for entry in allowlist.values():
        meta = map_metadata(client, entry.key, raw_dir)
        index += stage_song(client, meta, entry.split, raw_dir, out_dir)

    skipped: Counter[str] = Counter()
    songs = 0
    for meta in search_maps(client, search_params(cfg), raw_dir):
        if songs >= cfg.target_songs:
            break
        if meta["id"] in allowlist:
            continue
        reason = skip_reason(meta)
        if reason is None:
            split = song_split(meta["id"], cfg.heldout_fraction)
            try:
                index += stage_song(client, meta, split, raw_dir, out_dir)
                songs += 1
                continue
            except REJECTED as error:
                reason = f"{type(error).__name__}: {error}"
        skipped[reason] += 1
    return index, skipped
