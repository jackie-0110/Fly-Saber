"""Map parsing and normalization."""

import json
from pathlib import Path
from typing import Any

import pytest

from fly_saber.maps.allowlist import AllowlistEntry
from fly_saber.maps.chart import BpmChange, UnsupportedMapError
from fly_saber.maps.normalize import (
    beats_to_seconds,
    normalize_level,
    sample_clock,
    spawn_distance_m,
)

ALLOWED = AllowlistEntry(
    key="abc", title="Fixture", artist="Artist", license="CC BY 4.0", song_url="u", split="train"
)

INFO = {
    "_version": "2.0.0",
    "_songName": "Fixture",
    "_songAuthorName": "Artist",
    "_levelAuthorName": "Mapper",
    "_beatsPerMinute": 120,
    "_songTimeOffset": 0,
    "_difficultyBeatmapSets": [
        {
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [
                {
                    "_difficulty": "Easy",
                    "_beatmapFilename": "Easy.dat",
                    "_noteJumpMovementSpeed": 10,
                    "_noteJumpStartBeatOffset": 0,
                }
            ],
        }
    ],
}

V2_EASY = {
    "_version": "2.2.0",
    "_notes": [
        {"_time": 4, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
        {"_time": 2, "_lineIndex": 2, "_lineLayer": 1, "_type": 1, "_cutDirection": 4},
        {"_time": 6, "_lineIndex": 0, "_lineLayer": 2, "_type": 3, "_cutDirection": 0},
        {"_time": 8, "_lineIndex": 5, "_lineLayer": 0, "_type": 0, "_cutDirection": 0},
    ],
    "_obstacles": [{"_time": 3, "_lineIndex": 0, "_type": 1, "_duration": 1, "_width": 2}],
}

# Same chart as V2_EASY; zero-valued fields omitted as v3 writers do.
V3_EASY = {
    "version": "3.3.0",
    "colorNotes": [
        {"b": 4, "x": 1, "d": 1},
        {"b": 2, "x": 2, "y": 1, "c": 1, "d": 4},
        {"b": 8, "x": 5},
    ],
    "bombNotes": [{"b": 6, "y": 2}],
    "obstacles": [{"b": 3, "d": 1, "y": 2, "w": 2, "h": 3}],
}

# Same chart again in v4: objects index into shared data arrays.
V4_EASY = {
    "version": "4.0.0",
    "colorNotes": [{"b": 4}, {"b": 2, "i": 1}, {"b": 8, "i": 2}],
    "colorNotesData": [{"x": 1, "d": 1}, {"x": 2, "y": 1, "c": 1, "d": 4}, {"x": 5}],
    "bombNotes": [{"b": 6}],
    "bombNotesData": [{"y": 2}],
    "obstacles": [{"b": 3}],
    "obstaclesData": [{"d": 1, "y": 2, "w": 2, "h": 3}],
}

INFO_V4 = {
    "version": "4.0.1",
    "song": {"title": "Fixture", "author": "Artist"},
    "audio": {"songFilename": "song.ogg", "bpm": 120, "audioDataFilename": "AudioData.dat"},
    "difficultyBeatmaps": [
        {
            "characteristic": "Standard",
            "difficulty": "Easy",
            "beatmapAuthors": {"mappers": ["Mapper"]},
            "noteJumpMovementSpeed": 10,
            "noteJumpStartBeatOffset": 0,
            "beatmapDataFilename": "Easy.dat",
        }
    ],
}

# 120 BPM for 16 beats: 8 s of 44.1 kHz samples.
AUDIO_V4 = {"version": "4.0.0", "songFrequency": 44100, "bpmData": [{"ei": 352800, "eb": 16}]}


def write_level(
    root: Path,
    easy: dict[str, Any],
    info: dict[str, Any] = INFO,
    audio: dict[str, Any] | None = None,
) -> Path:
    root.mkdir()
    (root / "Info.dat").write_text(json.dumps(info))
    (root / "Easy.dat").write_text(json.dumps(easy))
    if audio is not None:
        (root / "AudioData.dat").write_text(json.dumps(audio))
    return root


def normalize(level: Path) -> dict[str, Any]:
    (normalized,) = normalize_level(
        level, key="abc", split="train", duration_sec=10.0, allowed=ALLOWED
    )
    return normalized


def test_every_format_normalizes_identically(tmp_path: Path) -> None:
    v2 = normalize(write_level(tmp_path / "v2", V2_EASY))
    assert normalize(write_level(tmp_path / "v3", V3_EASY)) == v2
    assert normalize(write_level(tmp_path / "v4", V4_EASY)) == v2
    assert normalize(write_level(tmp_path / "info4", V4_EASY, INFO_V4, AUDIO_V4)) == v2


def test_normalized_notes_are_sorted_in_seconds_with_bombs_and_off_grid_dropped(
    tmp_path: Path,
) -> None:
    normalized = normalize(write_level(tmp_path / "v2", V2_EASY))
    assert normalized["map_id"] == "abc_easy"
    fields = ("id", "time_sec", "lane", "layer", "color", "cut_direction", "is_bomb")
    assert [tuple(n[f] for f in fields) for n in normalized["notes"]] == [
        (0, 1.0, 2, 1, 1, 4, False),
        (1, 2.0, 1, 0, 0, 1, False),
        (2, 3.0, 0, 2, None, None, True),
    ]
    assert normalized["walls"] == [
        {"time_sec": 1.5, "duration_sec": 0.5, "lane": 0, "layer": 2, "width": 2, "height": 3}
    ]


def test_audio_url_only_for_allowlisted_maps(tmp_path: Path) -> None:
    level = write_level(tmp_path / "v2", V2_EASY)
    (unlisted,) = normalize_level(level, key="abc", split="train", duration_sec=1, allowed=None)
    assert unlisted["song"]["audio_url"] is None
    assert normalize(level)["song"]["audio_url"] == "/api/maps/abc_easy/audio"


def test_bpm_changes_apply_from_their_beat() -> None:
    changes = [BpmChange(beat=4, bpm=240)]
    assert beats_to_seconds(4, 120, changes) == pytest.approx(2.0)
    assert beats_to_seconds(8, 120, changes) == pytest.approx(3.0)


def test_spawn_distance_halves_jump_until_within_max() -> None:
    assert spawn_distance_m(njs=10, bpm=120, njs_offset=0) == pytest.approx(20 / 2)
    assert spawn_distance_m(njs=18, bpm=60, njs_offset=0) == pytest.approx(18.0)


def test_v2_bpm_changes_are_rejected(tmp_path: Path) -> None:
    easy = {**V2_EASY, "_BPMChanges": [{"_time": 4, "_BPM": 240}]}
    with pytest.raises(UnsupportedMapError):
        normalize(write_level(tmp_path / "v2", easy))


def test_v4_audio_regions_set_the_tempo() -> None:
    clock = sample_clock(
        {
            "songFrequency": 100,
            "bpmData": [{"ei": 200, "eb": 4}, {"si": 200, "ei": 300, "sb": 4, "eb": 8}],
        }
    )
    assert clock(2) == pytest.approx(1.0)
    assert clock(6) == pytest.approx(2.5)


def test_gapped_v4_audio_regions_are_rejected() -> None:
    with pytest.raises(UnsupportedMapError):
        sample_clock(
            {
                "songFrequency": 100,
                "bpmData": [{"ei": 200, "eb": 4}, {"si": 250, "ei": 300, "sb": 4, "eb": 8}],
            }
        )


def test_rotating_v4_charts_are_rejected(tmp_path: Path) -> None:
    easy = {**V4_EASY, "colorNotes": [{"b": 4, "r": 1}]}
    with pytest.raises(UnsupportedMapError):
        normalize(write_level(tmp_path / "v4", easy))
