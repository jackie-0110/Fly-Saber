"""Trained brains in the live server: bundles listed and loaded, ready, and play."""

import json
from pathlib import Path
from typing import Any

import jax
import pytest
from fastapi.testclient import TestClient

from fly_saber.config import load_config
from fly_saber.env import SaberConfig
from fly_saber.eye import EyeConfig
from server.app import create_app, serving_brains
from server.protocol import Brain
from tests.server.test_server import play_to_end, start, write_maps
from tests.training.test_brain_policy import model_config, write_subgraph
from training.bc import BCConfig
from training.brain_policy import build
from training.bundle import bundle, bundle_view_only
from training.checkpoints import save_checkpoint, subgraph_fingerprint


def mode(brain_id: str) -> Brain:
    return Brain(id=brain_id, label=brain_id.title(), description="test brain")


def write_bundles(directory: Path) -> Path:
    """bundles/ with a trained `real`, a view-only `shuffled_free`, a broken `shuffled_signs`,
    and no `shuffled_layers`."""
    subgraph = directory / "subgraph"
    subgraph.mkdir()
    write_subgraph(subgraph)
    counts = {"n_neurons": 4, "n_edges": 3, "n_synapses": 30}
    (subgraph / "meta.json").write_text(json.dumps(counts))
    (subgraph / "groups.json").write_text(json.dumps({"DN_all": [3]}))
    cfg = model_config(subgraph, [13, 5, 13, 5])
    saber, eye = load_config(SaberConfig, "saber"), load_config(EyeConfig, "eye")
    _, _, params = build(cfg, saber, eye, jax.random.PRNGKey(0))
    checkpoint = save_checkpoint(
        directory / "run" / "bc_1.npz",
        params,
        variant="real",
        stage="bc",
        step=1,
        configs={
            "model": cfg,
            "saber": saber,
            "eye": eye,
            "train": load_config(BCConfig, "train_bc"),
        },
        subgraph=subgraph_fingerprint(subgraph),
    )
    bundles = directory / "bundles"
    bundle(checkpoint, bundles / "real")
    bundle_view_only(subgraph, bundles / "shuffled_free", "rewired")
    bundle(checkpoint, bundles / "shuffled_signs")
    (bundles / "shuffled_signs" / "bc_1.npz").unlink()
    return bundles


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    directory = tmp_path_factory.mktemp("brains")
    write_maps(directory)
    bundles = write_bundles(directory)
    modes = [mode(m) for m in ("real", "shuffled_layers", "shuffled_signs", "shuffled_free")]
    return TestClient(create_app(directory, serving_brains(modes, bundles)))


def test_only_modes_with_a_loadable_trained_bundle_are_listed(client: TestClient) -> None:
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
