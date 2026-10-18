"""Trained brains in the live server: bundles listed and loaded, ready, and play."""

import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from fly_saber.config import load_config
from fly_saber.env import IDLE_CELL, SaberConfig
from fly_saber.eye import EyeConfig
from server.app import create_app, serving_brains
from server.protocol import Brain
from tests.server.test_server import play_to_end, receive, start, write_maps
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
    # Red saber idles while its descending neuron fires, and goes to cell 0 when it is silenced.
    params = params._replace(
        readout_w=jnp.zeros_like(params.readout_w).at[0, 0].set(-10.0),
        readout_b=jnp.zeros_like(params.readout_b).at[IDLE_CELL].set(1.0),
    )
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
        receive(ws)
        second = play_to_end(ws, 1)
    assert first[-1]["type"] == "ended"
    assert judged(first) == judged(second)


def sabers(ws: Any) -> dict[int, Any]:
    """Play the song in real time and return the saber poses by sim step. A late tick sends
    only its last step, so compare plays with `same_steps`."""
    ws.send_json({"type": "play", "song_t": 0.0})
    frames = {}
    while (message := receive(ws))["type"] != "ended":
        if message["type"] == "frame":
            frames[message["render_state"]["step"]] = message["render_state"]["sabers"]
    return frames


def same_steps(a: dict[int, Any], b: dict[int, Any]) -> bool:
    common = a.keys() & b.keys()
    assert len(common) > 10, "too few frames in common to compare"
    return all(a[k] == b[k] for k in common)


def play_with(ws: Any, interventions: dict[str, Any], eye: dict[str, Any]) -> dict[int, Any]:
    ws.send_json(
        {"type": "set_interventions", "slot": 0, "interventions": interventions, "eye": eye}
    )
    return sabers(ws)


def test_lesion_changes_play_and_lasts_through_reset(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        untouched = sabers(ws)
        lesioned = play_with(ws, {"lesion": ["DN_all"]}, {})
        ws.send_json({"type": "reset"})
        receive(ws)
        after_reset = sabers(ws)
    assert not same_steps(lesioned, untouched)
    assert same_steps(after_reset, lesioned)


def test_blinding_both_eyes_changes_play(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        untouched = sabers(ws)
        blind = play_with(ws, {}, {"blind_left": 1.0, "blind_right": 1.0})
    assert not same_steps(blind, untouched)


def test_new_start_clears_interventions(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        untouched = sabers(ws)
        play_with(ws, {"lesion": ["DN_all"]}, {})
        start(ws, ["real"])
        restarted = sabers(ws)
    assert same_steps(restarted, untouched)


@pytest.mark.parametrize(
    "message",
    [
        {"slot": 0, "interventions": {"lesion": ["no_such_group"]}},
        {"slot": 0, "interventions": {"mood": {"arousal": "high"}}},
        {"slot": 0, "eye": {"blind_left": 2.0}},
        {"slot": 1, "interventions": {"lesion": ["DN_all"]}},
        {"slot": 1, "eye": {"blur_deg": 5.0}},
    ],
)
def test_bad_interventions_get_error_and_socket_stays_open(
    client: TestClient, message: dict[str, Any]
) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real", "expert"])
        ws.send_json({"type": "set_interventions", **message})
        assert ws.receive_json()["type"] == "error"
        assert start(ws, ["real"])["type"] == "ready"


def activity(ws: Any) -> tuple[set[tuple[int, int]], list[tuple[int, int, int, bytes]]]:
    """Play the song in real time; return the (slot, step) of every frame, and every binary
    message as (kind, slot, step, payload)."""
    ws.send_json({"type": "play", "song_t": 0.0})
    frames, binary = set(), []
    while True:
        message = ws.receive()
        if "bytes" in message:
            kind, slot, reserved, step = struct.unpack_from("<BBHI", message["bytes"])
            assert reserved == 0
            binary.append((kind, slot, step, message["bytes"][8:]))
            continue
        decoded = json.loads(message["text"])
        if decoded["type"] == "frame":
            frames.add((decoded["slot"], decoded["render_state"]["step"]))
        if decoded["type"] == "ended" and decoded["slot"] == 0:
            return frames, binary


def test_brain_streams_activity_with_its_frames(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        ready = start(ws, ["real", "expert"])
        frames, binary = activity(ws)
    brain_frames = {step for slot, step in frames if slot == 0}
    assert {slot for _, slot, _, _ in binary} == {0}, "the expert has no brain to stream"
    types = [(step, payload) for kind, _, step, payload in binary if kind == 2]
    neurons = [(step, payload) for kind, _, step, payload in binary if kind == 1]
    assert [step for step, _ in types] == sorted(brain_frames)
    assert all(len(payload) == len(ready["cell_types"]) for _, payload in types)
    assert all(len(payload) == ready["n_neurons"] for _, payload in neurons)
    neuron_steps = [step for step, _ in neurons]
    assert set(neuron_steps) <= brain_frames
    assert all(b - a >= 2 for a, b in pairwise(neuron_steps)), "at most 15 Hz"
    assert len(neuron_steps) >= len(brain_frames) // 2 - 1


def test_lesioned_neuron_reads_below_rest(client: TestClient) -> None:
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real"])
        ws.send_json(
            {"type": "set_interventions", "slot": 0, "interventions": {"lesion": ["DN_all"]}}
        )
        _, binary = activity(ws)
    dn = 3  # groups.json: DN_all is neuron 3
    changes = [struct.unpack("4b", payload)[dn] for kind, _, _, payload in binary if kind == 1]
    assert changes and all(change < 0 for change in changes)


def test_eye_stream_matches_lattice_and_blindfolded_observation(client: TestClient) -> None:
    lattice = client.get("/api/eye_lattice?brain=real").json()
    with client.websocket_connect("/ws/session") as ws:
        start(ws, ["real", "expert"])
        ws.send_json(
            {"type": "set_interventions", "slot": 0, "eye": {"blind_left": 1, "blind_right": 1}}
        )
        frames, binary = activity(ws)
    images = [(slot, step, payload) for kind, slot, step, payload in binary if kind == 3]
    assert images
    for slot, step, payload in images:
        assert slot == 0 and (slot, step) in frames
        assert struct.unpack_from("<HH", payload) == (len(lattice["left"]), len(lattice["right"]))
        # Blinding removes contrast to background 0.5, it does not invent black objects.
        assert payload[4:] == bytes([128]) * (len(lattice["left"]) + len(lattice["right"]))
    assert all(b[1] - a[1] >= 2 for a, b in pairwise(images))
    assert len(images) >= len(frames) // 4 - 1


def test_interventions_recenter_the_readout(tmp_path: Path) -> None:
    """set_interventions re-centers the readout on the modified brain (PR #36); without it
    every lesion collapsed the trained fly to about 0% hits."""
    import numpy as np

    from server.saber_sim import brain_sims

    bundles = write_bundles(tmp_path)
    live = brain_sims(bundles / "real" / "bc_1.npz", bundles / "real" / "subgraph")
    _, untouched = live.interventions({})
    _, lesioned = live.interventions({"lesion": ["DN_all"]})
    trained = np.load(bundles / "real" / "bc_1.npz")[".dn_mean"]
    np.testing.assert_allclose(np.asarray(untouched.dn_mean), trained, rtol=1e-5)
    # The silenced DN rests at 0, so its baseline follows it down.
    assert float(lesioned.dn_mean[0]) == 0.0 < float(untouched.dn_mean[0])
