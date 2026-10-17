"""Normalize parsed maps to the shared schema (spec 8.2, docs/PROTOCOL.md section 1).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fly_saber.maps import parse_v2, parse_v3
from fly_saber.maps.allowlist import AllowlistEntry
from fly_saber.maps.chart import BpmChange, Chart, UnsupportedMapError

SCHEMA_VERSION = 1
N_LANES = 4
N_LAYERS = 3
N_CUT_DIRECTIONS = 9
TIME_DECIMALS = 6

# Beat Saber's own spawn rules (BeatmapObjectSpawnMovementData), so our notes
# appear where players of the map expect them.
BASE_HALF_JUMP_BEATS = 4.0
MAX_HALF_JUMP_DISTANCE_M = 18.0
MIN_HALF_JUMP_BEATS = 0.25
DEFAULT_NJS = {"Easy": 10.0, "Normal": 10.0, "Hard": 10.0, "Expert": 12.0, "ExpertPlus": 16.0}

PARSERS = {2: parse_v2.parse, 3: parse_v3.parse}


def load_dat(path: Path) -> dict[str, Any]:
    """Read a .dat JSON file; some editors write a UTF-8 BOM."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return data


def load_info(level_dir: Path) -> dict[str, Any]:
    """Read a level's Info.dat; the filename's case varies between maps."""
    return load_dat(next(p for p in level_dir.iterdir() if p.name.lower() == "info.dat"))


def format_major_version(dat: dict[str, Any]) -> int:
    version = dat.get("version") or dat.get("_version")
    if not isinstance(version, str):
        raise UnsupportedMapError("missing format version")
    return int(version.split(".")[0])


def beats_to_seconds(beat: float, base_bpm: float, changes: Iterable[BpmChange]) -> float:
    seconds, segment_start, bpm = 0.0, 0.0, base_bpm
    for change in sorted(changes, key=lambda c: c.beat):
        if change.beat >= beat:
            break
        seconds += (change.beat - segment_start) * 60.0 / bpm
        segment_start, bpm = change.beat, change.bpm
    return seconds + (beat - segment_start) * 60.0 / bpm


def spawn_distance_m(njs: float, bpm: float, njs_offset: float) -> float:
    beat_sec = 60.0 / bpm
    half_jump = BASE_HALF_JUMP_BEATS
    while njs * beat_sec * half_jump > MAX_HALF_JUMP_DISTANCE_M:
        half_jump /= 2
    half_jump = max(half_jump + njs_offset, MIN_HALF_JUMP_BEATS)
    return njs * beat_sec * half_jump


def normalize_chart(
    chart: Chart,
    *,
    bpm: float,
    map_id: str,
    split: str,
    song: dict[str, Any],
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    """Convert a beat-time chart to a normalized map dict, dropping off-grid objects."""

    def seconds(beat: float) -> float:
        return round(beats_to_seconds(beat, bpm, chart.bpm_changes), TIME_DECIMALS)

    on_grid = [
        n
        for n in chart.notes
        if 0 <= n.lane < N_LANES
        and 0 <= n.layer < N_LAYERS
        and (n.cut_direction is None or 0 <= n.cut_direction < N_CUT_DIRECTIONS)
    ]
    on_grid.sort(key=lambda n: (n.beat, n.lane, n.layer))
    notes = [
        {
            "id": i,
            "time_sec": seconds(n.beat),
            "lane": n.lane,
            "layer": n.layer,
            "color": n.color,
            "cut_direction": n.cut_direction,
            "is_bomb": n.color is None,
        }
        for i, n in enumerate(on_grid)
    ]
    walls = [
        {
            "time_sec": seconds(w.beat),
            "duration_sec": round(
                seconds(w.beat + w.duration_beats) - seconds(w.beat), TIME_DECIMALS
            ),
            "lane": w.lane,
            "layer": w.layer,
            "width": w.width,
            "height": w.height,
        }
        for w in sorted(chart.walls, key=lambda w: w.beat)
        if 0 <= w.lane < N_LANES and w.width > 0 and w.duration_beats > 0
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "map_id": map_id,
        "split": split,
        "song": song,
        "difficulty": difficulty,
        "notes": notes,
        "walls": walls,
    }


def normalize_level(
    level_dir: Path,
    *,
    key: str,
    split: str,
    duration_sec: float,
    allowed: AllowlistEntry | None,
) -> list[dict[str, Any]]:
    """Normalize every Standard difficulty of an extracted BeatSaver level.

    `allowed` is the level's allowlist entry; without one the map ships no audio.
    """
    info = load_info(level_dir)
    if format_major_version(info) != 2:
        raise UnsupportedMapError(f"Info.dat version {info.get('version')}")
    if info.get("_songTimeOffset", 0):
        raise UnsupportedMapError("nonzero _songTimeOffset")
    bpm = float(info["_beatsPerMinute"])

    maps = []
    for beatmap_set in info["_difficultyBeatmapSets"]:
        if beatmap_set["_beatmapCharacteristicName"] != "Standard":
            continue
        for beatmap in beatmap_set["_difficultyBeatmaps"]:
            name = beatmap["_difficulty"]
            dat = load_dat(level_dir / beatmap["_beatmapFilename"])
            parser = PARSERS.get(format_major_version(dat))
            if parser is None:
                raise UnsupportedMapError(f"{name} format {format_major_version(dat)}")
            map_id = f"{key}_{name}".lower()
            njs = float(beatmap["_noteJumpMovementSpeed"]) or DEFAULT_NJS[name]
            offset = float(beatmap["_noteJumpStartBeatOffset"])
            song = {
                "title": info["_songName"],
                "artist": info["_songAuthorName"],
                "mapper": info["_levelAuthorName"],
                "bpm": bpm,
                "duration_sec": duration_sec,
                "audio_url": f"/api/maps/{map_id}/audio" if allowed else None,
                "license": allowed.license if allowed else None,
                "attribution": allowed.attribution if allowed else None,
                "source_url": f"https://beatsaver.com/maps/{key}",
            }
            difficulty = {
                "name": name,
                "note_speed_mps": njs,
                "spawn_distance_m": round(spawn_distance_m(njs, bpm, offset), 4),
            }
            maps.append(
                normalize_chart(
                    parser(dat),
                    bpm=bpm,
                    map_id=map_id,
                    split=split,
                    song=song,
                    difficulty=difficulty,
                )
            )
    return maps
