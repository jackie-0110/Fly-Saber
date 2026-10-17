"""Checkpointed brains in the live server: listing, ready, and play."""

import json
from pathlib import Path
from typing import Any

import jax
import pytest
from fastapi.testclient import TestClient

from fly_saber.config import load_config
from fly_saber.env import SaberConfig
from fly_saber.eye import EyeConfig
from server.app import BrainSpec, create_app, serving_brains
from tests.server.test_server import play_to_end, start, write_maps
from tests.training.test_brain_policy import model_config, write_subgraph
from training.brain_policy import build
from training.checkpoints import save_checkpoint, subgraph_fingerprint


def spec(brain_id: str, checkpoint: Path) -> BrainSpec:
    return BrainSpec(brain_id, brain_id.title(), "test brain", str(checkpoint))


def write_checkpoint(directory: Path) -> Path:
    subgraph = directory / "subgraph"
    subgraph.mkdir()
    write_subgraph(subgraph)
    counts = {"n_neurons": 4, "n_edges": 3, "n_synapses": 30}
    (subgraph / "meta.json").write_text(json.dumps(counts))
    cfg = model_config(subgraph, [13, 5, 13, 5])
    saber = load_config(SaberConfig, "saber")
    _, _, params = build(cfg, saber, load_config(EyeConfig, "eye"), jax.random.PRNGKey(0))
    return save_checkpoint(
        directory / "real.npz",
        params,
        variant="real",
        stage="bc",
        step=1,
        configs={"model": cfg},
        subgraph=subgraph_fingerprint(subgraph),
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    directory = tmp_path_factory.mktemp("brains")
    write_maps(directory)
    checkpoint = write_checkpoint(directory)
    brains = serving_brains([spec("real", checkpoint), spec("rewired", directory / "none.npz")])
    return TestClient(create_app(directory, brains))


def test_only_brains_with_checkpoints_are_listed(client: TestClient) -> None:
    assert [b["id"] for b in client.get("/api/brains").json()] == ["expert", "real"]


def test_ready_reports_the_brains_neurons(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["expert", "real"])
    assert ready["n_neurons"] == 4
    assert ready["cell_types"] == ["DNp01", "R1-R6", "R7p"]


def test_brain_plays_the_song_and_replays_identically(client: TestClient) -> None:
    def judged(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [m for m in messages if m["type"] in ("event", "ended")]

    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        first = play_to_end(ws, 1)
        ws.send_json({"type": "reset"})
        ws.receive_json()
        second = play_to_end(ws, 1)
    assert first[-1]["type"] == "ended"
    assert judged(first) == judged(second)
