"""REST and WS /ws/session against a tiny fixture map and the fake sim."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from flyarcade.server.app import create_app

AUDIO_BYTES = b"OggS fixture"


def make_map(map_id: str) -> dict[str, Any]:
    notes = [
        (0.2, 0, 0, 0, 1, False),
        (0.4, 2, 1, 1, 5, False),
        (0.5, 1, 0, None, None, True),
        (0.6, 3, 2, 0, 8, False),
    ]
    return {
        "schema_version": 1,
        "map_id": map_id,
        "split": "train",
        "song": {
            "title": "Fixture",
            "artist": "Tester",
            "mapper": "mapper",
            "bpm": 120.0,
            "duration_sec": 1.0,
            "audio_url": f"/api/maps/{map_id}/audio",
            "license": "CC BY 4.0",
            "source_url": "https://example.com",
        },
        "difficulty": {"name": "Hard", "note_speed_mps": 10.0, "spawn_distance_m": 12.0},
        "notes": [
            {
                "id": i,
                "time_sec": t,
                "lane": lane,
                "layer": layer,
                "color": color,
                "cut_direction": cut,
                "is_bomb": bomb,
            }
            for i, (t, lane, layer, color, cut, bomb) in enumerate(notes)
        ],
        "walls": [],
    }


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    (tmp_path / "normalized").mkdir()
    (tmp_path / "audio").mkdir()
    for map_id in ("abc12_hard", "zzz99_easy"):
        (tmp_path / "normalized" / f"{map_id}.json").write_text(json.dumps(make_map(map_id)))
    (tmp_path / "audio" / "abc12.ogg").write_bytes(AUDIO_BYTES)
    (tmp_path / "audio" / "zzz99.ogg").write_bytes(AUDIO_BYTES)
    (tmp_path / "demo_allowlist.json").write_text(json.dumps(["abc12"]))
    return TestClient(create_app(tmp_path))


def test_games(client: TestClient) -> None:
    assert client.get("/api/games").json() == [{"id": "saber", "label": "Fly Saber"}]


def test_brains_have_id_label_description(client: TestClient) -> None:
    brains = client.get("/api/brains").json()
    assert brains
    assert all(set(b) == {"id", "label", "description"} for b in brains)


def test_map_list_has_only_allowlisted_summaries(client: TestClient) -> None:
    assert client.get("/api/maps").json() == [
        {
            "map_id": "abc12_hard",
            "split": "train",
            "song": {
                "title": "Fixture",
                "artist": "Tester",
                "mapper": "mapper",
                "license": "CC BY 4.0",
            },
            "difficulty": {"name": "Hard"},
            "note_count": 4,
        }
    ]


def test_map_detail_returns_full_map(client: TestClient) -> None:
    assert client.get("/api/maps/zzz99_easy").json() == make_map("zzz99_easy")


@pytest.mark.parametrize("map_id", ["nope1_hard", "..%2Fdemo_allowlist"])
def test_unknown_map_is_404(client: TestClient, map_id: str) -> None:
    assert client.get(f"/api/maps/{map_id}").status_code == 404


def test_audio_served_only_when_allowlisted(client: TestClient) -> None:
    allowed = client.get("/api/maps/abc12_hard/audio")
    assert allowed.status_code == 200
    assert allowed.content == AUDIO_BYTES
    assert client.get("/api/maps/zzz99_easy/audio").status_code == 404


def start(ws: WebSocketTestSession, brains: list[str]) -> dict[str, Any]:
    ws.send_json(
        {"type": "start", "game": "saber", "brains": brains, "map_id": "abc12_hard", "seed": 7}
    )
    ready: dict[str, Any] = ws.receive_json()
    return ready


def play_to_end(ws: WebSocketTestSession, n_slots: int) -> list[dict[str, Any]]:
    """Play from 0, jump the clock past the song end, and collect messages until every slot ends."""
    ws.send_json({"type": "play", "song_t": 0.0})
    ws.send_json({"type": "clock", "song_t": 5.0})
    messages: list[dict[str, Any]] = []
    while sum(m["type"] == "ended" for m in messages) < n_slots:
        messages.append(ws.receive_json())
    return messages


def test_start_answers_ready(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["real"])
    assert ready["type"] == "ready"
    assert ready["map_id"] == "abc12_hard"
    assert ready["slots"] == [{"slot": 0, "brain": "real"}]
    assert ready["step_hz"] == 30


def test_play_judges_every_note_once_and_ends(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        messages = play_to_end(ws, 1)
    events = [m for m in messages if m["type"] == "event"]
    assert sorted(e["note_id"] for e in events) == [0, 1, 3]
    frames = [m for m in messages if m["type"] == "frame"]
    assert frames
    assert frames[-1]["render_state"]["t"] >= 1.0
    score = messages[-1]["score"]
    assert score["hits"] + score["misses"] == 3


def test_versus_runs_both_slots(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["real", "scrambled"])
        messages = play_to_end(ws, 2)
    assert [s["brain"] for s in ready["slots"]] == ["real", "scrambled"]
    assert {m["slot"] for m in messages if m["type"] == "ended"} == {0, 1}


def test_reset_returns_to_ready_and_can_replay(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        first = play_to_end(ws, 1)[-1]
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "ready"
        second = play_to_end(ws, 1)[-1]
    assert first == second


@pytest.mark.parametrize(
    "message",
    [
        {"type": "warp"},
        {"type": "play", "song_t": 0.0},
        {"type": "start", "game": "saber", "brains": ["real"], "map_id": "nope1_hard", "seed": 0},
    ],
)
def test_bad_message_gets_error_and_socket_stays_open(
    client: TestClient, message: dict[str, Any]
) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ws.send_json(message)
        assert ws.receive_json()["type"] == "error"
        assert start(ws, ["real"])["type"] == "ready"
