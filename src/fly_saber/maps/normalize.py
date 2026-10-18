"""Normalize parsed maps to the shared schema (spec 8.2, docs/PROTOCOL.md section 1).

Owner: Person 2. See docs/project/FLY_SABER_HANDOFF.md for the spec.
"""

import json
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

from fly_saber.maps import parse_v2, parse_v3, parse_v4
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

PARSERS = {2: parse_v2.parse, 3: parse_v3.parse, 4: parse_v4.parse}
# Beats and seconds; the browser's sampleClock uses the same tolerance.
REGION_TOLERANCE = 0.001


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


def sample_clock(audio: dict[str, Any]) -> Callable[[float], float]:
    """Beat to seconds from a v4 audio data file, whose BPM regions map beats to samples."""
    frequency = float(audio.get("songFrequency", 0))
    if frequency <= 0:
        raise UnsupportedMapError("audio data has no song frequency")
    regions = sorted(
        (
            float(r.get("sb", 0)),
            float(r.get("eb", 0)),
            float(r.get("si", 0)) / frequency,
            float(r.get("ei", 0)) / frequency,
        )
        for r in audio.get("bpmData", [])
    )
    if (
        not regions
        or any(
            end_beat <= start_beat or end <= start for start_beat, end_beat, start, end in regions
        )
        or any(
            abs(b[0] - a[1]) > REGION_TOLERANCE or abs(b[2] - a[3]) > REGION_TOLERANCE
            for a, b in pairwise(regions)
        )
    ):
        raise UnsupportedMapError("invalid audio timing regions")
    starts = [r[0] for r in regions]

    def seconds(beat: float) -> float:
        start_beat, end_beat, start, end = regions[max(bisect_right(starts, beat) - 1, 0)]
        return start + (beat - start_beat) / (end_beat - start_beat) * (end - start)

    return seconds


@dataclass(frozen=True)
class Beatmap:
    difficulty: str
    filename: str
    njs: float
    njs_offset: float


@dataclass(frozen=True)
class LevelInfo:
    title: str
    artist: str
    mapper: str
    song_filename: str
    bpm: float
    beatmaps: list[Beatmap]
    """Standard difficulties only."""
    clock: Callable[[float], float] | None
    """Beat to seconds for the whole level (v4), or None to use each chart's BPM changes."""


def read_info_v2(info: dict[str, Any]) -> LevelInfo:
    if info.get("_songTimeOffset", 0):
        raise UnsupportedMapError("nonzero _songTimeOffset")
    beatmaps = [
        Beatmap(
            b["_difficulty"],
            b["_beatmapFilename"],
            float(b["_noteJumpMovementSpeed"]),
            float(b["_noteJumpStartBeatOffset"]),
        )
        for s in info["_difficultyBeatmapSets"]
        if s["_beatmapCharacteristicName"] == "Standard"
        for b in s["_difficultyBeatmaps"]
    ]
    return LevelInfo(
        info["_songName"],
        info["_songAuthorName"],
        info["_levelAuthorName"],
        info["_songFilename"],
        float(info["_beatsPerMinute"]),
        beatmaps,
        None,
    )


def read_info_v4(info: dict[str, Any], level_dir: Path) -> LevelInfo:
    standard = [b for b in info["difficultyBeatmaps"] if b["characteristic"] == "Standard"]
    mappers = dict.fromkeys(
        m.strip()
        for b in standard
        for m in b.get("beatmapAuthors", {}).get("mappers", [])
        if isinstance(m, str) and m.strip()
    )
    beatmaps = [
        Beatmap(
            b["difficulty"],
            b["beatmapDataFilename"],
            float(b.get("noteJumpMovementSpeed", 0)),
            float(b.get("noteJumpStartBeatOffset", 0)),
        )
        for b in standard
    ]
    audio = info["audio"]
    return LevelInfo(
        info["song"]["title"],
        info["song"]["author"],
        ", ".join(mappers),
        audio["songFilename"],
        float(audio["bpm"]),
        beatmaps,
        sample_clock(load_dat(level_dir / audio["audioDataFilename"])),
    )


def read_info(level_dir: Path) -> LevelInfo:
    """Read a level's Info.dat, v2 or v4."""
    info = load_info(level_dir)
    version = format_major_version(info)
    if version == 2:
        return read_info_v2(info)
    if version == 4:
        return read_info_v4(info, level_dir)
    raise UnsupportedMapError(f"Info.dat version {version}")


def normalize_chart(
    chart: Chart,
    *,
    beat_seconds: Callable[[float], float],
    map_id: str,
    split: str,
    song: dict[str, Any],
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    """Convert a beat-time chart to a normalized map dict, dropping off-grid objects."""

    def seconds(beat: float) -> float:
        return round(beat_seconds(beat), TIME_DECIMALS)

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
    level = read_info(level_dir)
    maps = []
    for beatmap in level.beatmaps:
        dat = load_dat(level_dir / beatmap.filename)
        parser = PARSERS.get(format_major_version(dat))
        if parser is None:
            raise UnsupportedMapError(f"{beatmap.difficulty} format {format_major_version(dat)}")
        chart = parser(dat)
        map_id = f"{key}_{beatmap.difficulty}".lower()
        njs = beatmap.njs or DEFAULT_NJS[beatmap.difficulty]
        song = {
            "title": level.title,
            "artist": level.artist,
            "mapper": level.mapper,
            "bpm": level.bpm,
            "duration_sec": duration_sec,
            "audio_url": f"/api/maps/{map_id}/audio" if allowed else None,
            "license": allowed.license if allowed else None,
            "attribution": allowed.attribution if allowed else None,
            "source_url": f"https://beatsaver.com/maps/{key}",
        }
        difficulty = {
            "name": beatmap.difficulty,
            "note_speed_mps": njs,
            "spawn_distance_m": round(spawn_distance_m(njs, level.bpm, beatmap.njs_offset), 4),
        }
        maps.append(
            normalize_chart(
                chart,
                beat_seconds=level.clock
                or partial(beats_to_seconds, base_bpm=level.bpm, changes=chart.bpm_changes),
                map_id=map_id,
                split=split,
                song=song,
                difficulty=difficulty,
            )
        )
    return maps
