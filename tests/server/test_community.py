"""On-demand community maps: fetched once, charts only, playable by key, errors mapped."""

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.community import CommunityMaps
from tests.fly_saber.test_training_set import FakeBeatSaver, info_with, level_zip, meta
from tests.server.test_server import EXPERT_BRAIN, play_to_end, start, write_maps


@pytest.fixture
def beatsaver() -> FakeBeatSaver:
    return FakeBeatSaver(
        [[meta("c0ffee", "Easy", "Hard")]], {"c0ffee": level_zip(info_with("Easy", "Hard"))}
    )


def make_client(tmp_path: Path, beatsaver: FakeBeatSaver) -> TestClient:
    write_maps(tmp_path)
    community = CommunityMaps(tmp_path / "community", beatsaver.client)
    return TestClient(create_app(tmp_path, [EXPERT_BRAIN], community=community))


def test_import_returns_map_ids_and_downloads_once(
    tmp_path: Path, beatsaver: FakeBeatSaver
) -> None:
    client = make_client(tmp_path, beatsaver)
    for _ in range(2):
        response = client.post("/api/community/C0FFEE")
        assert response.status_code == 200
        assert response.json() == ["c0ffee_easy", "c0ffee_hard"]
    assert beatsaver.paths == ["/maps/id/c0ffee", "/c0ffee.zip"]
    detail = client.get("/api/maps/c0ffee_hard").json()
    assert detail["split"] == "community"
    assert detail["song"]["audio_url"] is None


def test_imported_map_has_no_audio_and_is_not_listed(
    tmp_path: Path, beatsaver: FakeBeatSaver
) -> None:
    client = make_client(tmp_path, beatsaver)
    client.post("/api/community/c0ffee")
    assert client.get("/api/maps/c0ffee_easy/audio").status_code == 404
    assert "c0ffee_easy" not in {m["map_id"] for m in client.get("/api/maps").json()}
    assert not list((tmp_path / "community" / "raw").rglob("*.egg"))


def test_imported_map_plays_live(tmp_path: Path, beatsaver: FakeBeatSaver) -> None:
    client = make_client(tmp_path, beatsaver)
    client.post("/api/community/c0ffee")
    with client.websocket_connect("/ws/session") as ws:
        assert start(ws, ["expert"], map_id="c0ffee_easy")["type"] == "ready"
        assert play_to_end(ws, 1)


@pytest.mark.parametrize(("key", "status"), [("not-hex", 400), ("123456789", 400), ("dead", 404)])
def test_bad_or_unknown_key(tmp_path: Path, key: str, status: int) -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    write_maps(tmp_path)
    community = CommunityMaps(
        tmp_path / "community", lambda: httpx.Client(transport=httpx.MockTransport(missing))
    )
    client = TestClient(create_app(tmp_path, [EXPERT_BRAIN], community=community))
    assert client.post(f"/api/community/{key}").status_code == status


def test_unsupported_map_is_422(tmp_path: Path) -> None:
    info = info_with("Easy", _version="3.0.0")
    beatsaver = FakeBeatSaver([[meta("abc1", "Easy")]], {"abc1": level_zip(info)})
    client = make_client(tmp_path, beatsaver)
    response = client.post("/api/community/abc1")
    assert response.status_code == 422
    assert "cannot play" in response.json()["detail"]


def test_v4_map_missing_its_audio_data_is_422(tmp_path: Path) -> None:
    info = {
        "version": "4.0.1",
        "song": {"title": "T", "author": "A"},
        "audio": {"songFilename": "song.ogg", "bpm": 120, "audioDataFilename": "AudioData.dat"},
        "difficultyBeatmaps": [
            {
                "characteristic": "Standard",
                "difficulty": "Easy",
                "beatmapDataFilename": "Easy.dat",
            }
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Info.dat", json.dumps(info))
        archive.writestr("Easy.dat", "{}")
    beatsaver = FakeBeatSaver([[meta("abc2", "Easy")]], {"abc2": buffer.getvalue()})
    response = make_client(tmp_path, beatsaver).post("/api/community/abc2")
    assert response.status_code == 422
