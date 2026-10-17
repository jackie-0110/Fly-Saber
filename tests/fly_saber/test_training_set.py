"""BeatSaver training-set scrape against a mocked API: paging, filters, splits, no audio."""

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from fly_saber.maps.allowlist import AllowlistEntry
from fly_saber.maps.beatsaver_client import search_maps
from fly_saber.maps.training_set import (
    TrainingMapsConfig,
    build_training_set,
    skip_reason,
    song_split,
)
from tests.fly_saber.test_maps import INFO, V2_EASY

CDN = "https://cdn.test"


def info_with(*names: str, **overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = copy.deepcopy(INFO)
    (beatmap,) = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"] = [
        {**beatmap, "_difficulty": name, "_beatmapFilename": f"{name}.dat"} for name in names
    ]
    return {**info, **overrides}


def level_zip(info: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Info.dat", json.dumps(info))
        for beatmap in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]:
            archive.writestr(beatmap["_beatmapFilename"], json.dumps(V2_EASY))
        archive.writestr("song.egg", b"OggS audio")
        archive.writestr("cover.jpg", b"art")
    return buffer.getvalue()


def meta(key: str, *difficulties: str, **overrides: Any) -> dict[str, Any]:
    diffs = [{"characteristic": "Standard", "difficulty": d} for d in difficulties or ("Easy",)]
    version = {"diffs": diffs, "downloadURL": f"{CDN}/{key}.zip"}
    return {
        "id": key,
        "metadata": {"duration": 10},
        "automapper": False,
        "declaredAi": "None",
        "versions": [version],
        **overrides,
    }


class FakeBeatSaver:
    """Serves search pages, map metadata, and level zips; records every request path."""

    def __init__(self, pages: list[list[dict[str, Any]]], zips: dict[str, bytes]) -> None:
        self.pages, self.zips = pages, zips
        self.paths: list[str] = []
        self.metas = {m["id"]: m for page in pages for m in page}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.startswith("/search/text/"):
            page = int(path.rsplit("/", 1)[1])
            docs = self.pages[page] if page < len(self.pages) else []
            return httpx.Response(200, json={"docs": docs, "info": {"pages": len(self.pages)}})
        if path.startswith("/maps/id/"):
            return httpx.Response(200, json=self.metas[path.rsplit("/", 1)[1]])
        return httpx.Response(200, content=self.zips[path.strip("/").removesuffix(".zip")])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def config(tmp_path: Path, target_songs: int = 10) -> TrainingMapsConfig:
    return TrainingMapsConfig(
        raw_dir=str(tmp_path / "raw"),
        out_dir=str(tmp_path / "out"),
        target_songs=target_songs,
        min_rating=0.8,
        min_upvotes=0,
        page_size=2,
        heldout_fraction=0.15,
        request_interval_sec=0.0,
    )


def test_search_stops_after_the_last_page(tmp_path: Path) -> None:
    api = FakeBeatSaver([[meta("a1"), meta("a2")], [meta("a3")]], {})
    keys = [m["id"] for m in search_maps(api.client(), {}, tmp_path)]
    assert keys == ["a1", "a2", "a3"]
    assert api.paths == ["/search/text/0", "/search/text/1"]


def test_search_reruns_from_cache_without_requests(tmp_path: Path) -> None:
    api = FakeBeatSaver([[meta("a1")]], {})
    list(search_maps(api.client(), {"q": "x"}, tmp_path))
    offline = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert [m["id"] for m in search_maps(offline, {"q": "x"}, tmp_path)] == ["a1"]


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (meta("k", automapper=True), "automapped"),
        (meta("k", declaredAi="Uploader"), "automapped"),
        (meta("k", "Expert", "ExpertPlus"), "no Easy, Normal, or Hard"),
        (meta("k", "Easy", "Hard"), None),
    ],
)
def test_skip_reason_filters_automapped_maps(result: dict[str, Any], reason: str | None) -> None:
    assert skip_reason(result) == reason


@pytest.mark.parametrize("flag", ["chroma", "ne", "me", "vivify"])
def test_skip_reason_filters_modded_maps(flag: str) -> None:
    modded = meta("k", "Easy", "Hard")
    modded["versions"][0]["diffs"][1][flag] = True
    assert skip_reason(modded) == "uses mods"


def test_rejected_songs_are_skipped_with_their_reason(tmp_path: Path) -> None:
    api = FakeBeatSaver(
        [[meta("bad"), meta("good")]],
        {"bad": level_zip(info_with("Easy", _songTimeOffset=1.5)), "good": level_zip(INFO)},
    )
    index, skipped = build_training_set(api.client(), config(tmp_path), {})
    assert [row["map_id"] for row in index] == ["good_easy"]
    assert skipped == {"UnsupportedMapError: nonzero _songTimeOffset": 1}
    assert [p.name for p in (tmp_path / "out" / "normalized").iterdir()] == ["good_easy.json"]


def test_no_audio_is_kept(tmp_path: Path) -> None:
    api = FakeBeatSaver([[meta("song")]], {"song": level_zip(INFO)})
    build_training_set(api.client(), config(tmp_path), {})
    kept = {p.name for p in (tmp_path / "raw" / "song").iterdir()}
    assert kept == {"Info.dat", "Easy.dat", "beatsaver.json"}
    normalized = json.loads((tmp_path / "out" / "normalized" / "song_easy.json").read_text())
    assert normalized["song"]["audio_url"] is None


def test_split_is_per_song_and_deterministic(tmp_path: Path) -> None:
    api = FakeBeatSaver(
        [[meta("song", "Easy", "Hard")]], {"song": level_zip(info_with("Easy", "Hard"))}
    )
    index, _ = build_training_set(api.client(), config(tmp_path), {})
    assert [row["split"] for row in index] == [song_split("song", 0.15)] * 2
    heldout = [song_split(f"{i:x}", 0.15) for i in range(2000)].count("heldout") / 2000
    assert 0.12 < heldout < 0.18


def test_demo_songs_keep_their_allowlist_split_and_no_audio(tmp_path: Path) -> None:
    api = FakeBeatSaver(
        [[meta("demo"), meta("other")]], {"demo": level_zip(INFO), "other": level_zip(INFO)}
    )
    entry = AllowlistEntry(
        key="demo", title="t", artist="a", license="CC BY 4.0", song_url="u", split="heldout"
    )
    index, _ = build_training_set(api.client(), config(tmp_path), {"demo": entry})
    demo = [row for row in index if row["key"] == "demo"]
    assert [row["split"] for row in demo] == ["heldout"]
    assert {row["key"] for row in index} == {"demo", "other"}
    normalized = json.loads((tmp_path / "out" / "normalized" / "demo_easy.json").read_text())
    assert normalized["song"]["audio_url"] is None
