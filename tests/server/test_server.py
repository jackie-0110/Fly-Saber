"""REST and WS /ws/session against a tiny fixture map and the real env played by the expert."""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from server.app import EXPERT, create_app
from server.protocol import Brain, RenderState
from server.saber_sim import MAX_NOTES, expert_sims
from server.session import LiveBrain, Sim, StepOutput

AUDIO_BYTES = b"OggS fixture"
EXPERT_BRAIN = LiveBrain(EXPERT, expert_sims())


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


def write_maps(tmp_path: Path) -> None:
    """Two fixture maps; only abc12_hard has allowlisted audio."""
    (tmp_path / "normalized").mkdir()
    (tmp_path / "audio").mkdir()
    for map_id in ("abc12_hard", "zzz99_easy"):
        (tmp_path / "normalized" / f"{map_id}.json").write_text(json.dumps(make_map(map_id)))
    (tmp_path / "audio" / "abc12.ogg").write_bytes(AUDIO_BYTES)
    (tmp_path / "audio" / "zzz99.ogg").write_bytes(AUDIO_BYTES)
    allowlist = [
        {
            "key": "abc12",
            "title": "Fixture",
            "artist": "Tester",
            "license": "CC BY 4.0",
            "song_url": "https://example.com",
            "split": "train",
        }
    ]
    (tmp_path / "demo_allowlist.json").write_text(json.dumps(allowlist))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    write_maps(tmp_path)
    return TestClient(create_app(tmp_path, [EXPERT_BRAIN]))


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


def start(
    ws: WebSocketTestSession, brains: list[str], map_id: str = "abc12_hard"
) -> dict[str, Any]:
    ws.send_json({"type": "start", "game": "saber", "brains": brains, "map_id": map_id, "seed": 7})
    ready: dict[str, Any] = ws.receive_json()
    return ready


def receive(ws: WebSocketTestSession) -> dict[str, Any]:
    """The next JSON message, skipping binary activity."""
    message = ws.receive()
    while "bytes" in message:
        message = ws.receive()
    decoded: dict[str, Any] = json.loads(message["text"])
    return decoded


def play_to_end(ws: WebSocketTestSession, n_slots: int) -> list[dict[str, Any]]:
    """Play from 0, jump the clock past the song end, and collect messages until every slot ends."""
    ws.send_json({"type": "play", "song_t": 0.0})
    ws.send_json({"type": "clock", "song_t": 5.0})
    messages: list[dict[str, Any]] = []
    while sum(m["type"] == "ended" for m in messages) < n_slots:
        messages.append(receive(ws))
    return messages


def test_start_answers_ready(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["expert"])
    assert ready["type"] == "ready"
    assert ready["map_id"] == "abc12_hard"
    assert ready["slots"] == [{"slot": 0, "brain": "expert"}]
    assert ready["step_hz"] == 30


def test_play_judges_every_note_once_and_ends(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["expert"])
        messages = play_to_end(ws, 1)
    events = [m for m in messages if m["type"] == "event"]
    assert sorted(e["note_id"] for e in events) == [0, 1, 3]
    assert {e["kind"] for e in events} == {"hit"}
    frames = [m for m in messages if m["type"] == "frame"]
    assert frames
    last_note, window = 0.6, 0.1
    assert frames[-1]["render_state"]["t"] > last_note + window
    score = messages[-1]["score"]
    assert score["hits"] + score["misses"] == 3


def test_versus_runs_both_slots(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["expert", "expert"])
        messages = play_to_end(ws, 2)
    assert [s["brain"] for s in ready["slots"]] == ["expert", "expert"]
    assert {m["slot"] for m in messages if m["type"] == "ended"} == {0, 1}


def test_reset_returns_to_ready_and_can_replay(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["expert"])
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
        {"type": "start", "game": "saber", "brains": ["expert"], "map_id": "nope1_hard", "seed": 0},
    ],
)
def test_bad_message_gets_error_and_socket_stays_open(
    client: TestClient, message: dict[str, Any]
) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ws.send_json(message)
        assert ws.receive_json()["type"] == "error"
        assert start(ws, ["expert"])["type"] == "ready"


@dataclass
class SlowSim:
    """An expert sim whose every step blocks, like a big brain on a slow CPU."""

    sim: Sim
    step_sec: float

    def reset(self, song_t: float) -> RenderState:
        return self.sim.reset(song_t)

    def step(self) -> StepOutput:
        time.sleep(self.step_sec)
        return self.sim.step()

    def set_interventions(self, controls: Mapping[str, Any], eye: Mapping[str, Any]) -> None:
        self.sim.set_interventions(controls, eye)

    def activity(self) -> tuple[bytes, bytes] | None:
        return self.sim.activity()

    def eye_image(self) -> bytes | None:
        return self.sim.eye_image()


class SlowSims:
    n_neurons = 0
    cell_types: list[str] = []

    def eye_lattice(self) -> dict[str, list[list[float]]]:
        return EXPERT_BRAIN.make_sim.eye_lattice()

    def __call__(self, map_data: dict[str, Any], seed: int) -> SlowSim:
        return SlowSim(EXPERT_BRAIN.make_sim(map_data, seed), step_sec=0.5)


def test_session_answers_while_its_sim_is_stepping(tmp_path: Path) -> None:
    write_maps(tmp_path)
    slow = LiveBrain(Brain(id="expert", label="Slow", description="test"), SlowSims())
    with TestClient(create_app(tmp_path, [slow])).websocket_connect("/ws/session") as ws:
        start(ws, ["expert"])
        ws.send_json({"type": "play", "song_t": 0.0})
        ws.send_json({"type": "clock", "song_t": 5.0})  # seconds of slow steps to catch up
        time.sleep(0.2)
        sent = time.monotonic()
        ws.send_json({"type": "warp"})
        assert ws.receive_json()["type"] == "error"
        assert time.monotonic() - sent < 0.25


def test_start_past_max_sessions_is_refused_until_one_leaves(tmp_path: Path) -> None:
    write_maps(tmp_path)
    client = TestClient(create_app(tmp_path, [EXPERT_BRAIN], max_sessions=1))
    with client.websocket_connect("/ws/session") as second:
        with client.websocket_connect("/ws/session") as first:
            assert start(first, ["expert"])["type"] == "ready"
            refused = start(second, ["expert"])
            assert refused["type"] == "error"
            assert "full" in refused["message"]
        for _ in range(50):  # the first seat frees once the server sees the disconnect
            if start(second, ["expert"])["type"] == "ready":
                return
            time.sleep(0.05)
        pytest.fail("the seat was never freed")


def write_map(tmp_path: Path, map_id: str, n_notes: int) -> None:
    data = make_map(map_id)
    note = data["notes"][0]
    data["notes"] = [{**note, "id": i, "time_sec": 0.2 + 0.01 * i} for i in range(n_notes)]
    (tmp_path / "normalized" / f"{map_id}.json").write_text(json.dumps(data))


def test_no_map_size_compiles_after_startup(tmp_path: Path) -> None:
    write_maps(tmp_path)
    write_map(tmp_path, "big12_hard", 100)
    app = create_app(tmp_path, [LiveBrain(EXPERT, expert_sims())])
    compiles: list[str] = []

    def on_event(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(on_event)
    try:
        with TestClient(app).websocket_connect("/ws/session") as ws:
            assert start(ws, ["expert"], "big12_hard")["type"] == "ready"
            ws.send_json({"type": "play", "song_t": 0.0})
            while receive(ws)["type"] != "frame":
                pass
    finally:
        jax.monitoring.unregister_event_duration_listener(on_event)
    assert compiles == []


def test_map_over_the_note_limit_gets_error_and_socket_stays_open(tmp_path: Path) -> None:
    write_maps(tmp_path)
    write_map(tmp_path, "huge1_hard", MAX_NOTES + 1)
    with TestClient(create_app(tmp_path, [EXPERT_BRAIN])).websocket_connect("/ws/session") as ws:
        refused = start(ws, ["expert"], "huge1_hard")
        assert refused["type"] == "error"
        assert str(MAX_NOTES) in refused["message"]
        assert start(ws, ["expert"])["type"] == "ready"
