"""Client messages reject invalid controls before reaching the live simulation."""

import struct
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fly_saber.env import SaberEnv
from server.app import create_app
from server.protocol import EYE_IMAGE, binary_message, client_message_adapter
from server.saber_sim import LiveSaber, expert_sims
from server.session import LiveBrain
from tests.server.test_server import EXPERT_BRAIN, write_maps


@pytest.mark.parametrize(
    "message",
    [
        {"type": "start", "game": "saber", "brains": [], "map_id": "abc_easy", "seed": 0},
        {
            "type": "start",
            "game": "saber",
            "brains": ["real", "real", "real"],
            "map_id": "abc_easy",
            "seed": 0,
        },
        {"type": "human_action", "color": 2, "cell": 0, "cut": 0},
        {"type": "human_action", "color": 0, "cell": 12, "cut": 0},
        {"type": "human_action", "color": 0, "cell": -1, "cut": 0},
        {"type": "human_action", "color": 0, "cell": 0, "cut": 4},
        {"type": "set_interventions", "slot": 2, "interventions": {}},
        {"type": "reset", "unexpected": True},
        {"type": "unknown"},
    ],
)
def test_invalid_client_message_is_rejected(message: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(message)


@pytest.mark.parametrize("cell, cut", [(None, None), (0, 0), (11, 3)])
def test_human_action_accepts_idle_and_grid_boundaries(cell: int | None, cut: int | None) -> None:
    payload = {"type": "human_action", "color": 1, "cell": cell, "cut": cut}
    assert client_message_adapter.validate_python(payload).model_dump() == payload


def test_eye_image_layout_and_lattice_share_left_first_order(tmp_path: Path) -> None:
    """Interleaved columns and overlapping eyes must not be split by azimuth."""
    from server.protocol import Brain

    base = expert_sims()
    directions = np.radians([[10.0, 20.0], [-10.0, 30.0], [-20.0, 40.0]])
    env = SaberEnv.create(
        base.env.cfg, base.env.eye_cfg, directions, np.array([False, True, False])
    )
    live = LiveSaber(
        env, lambda s, o, c, iv, k: (jnp.zeros((2, 2), jnp.int32), c, o[0]), (), n_neurons=3
    )
    write_maps(tmp_path)
    brain = LiveBrain(Brain(id="real", label="Fixture", description="test"), live)
    client = TestClient(create_app(tmp_path, [EXPERT_BRAIN, brain]))
    angles = client.get("/api/eye_lattice").json()
    np.testing.assert_allclose(angles["left"], [[-10, 30]], atol=1e-5)
    np.testing.assert_allclose(angles["right"], [[10, 20], [-20, 40]], atol=1e-5)
    assert client.get("/api/eye_lattice?brain=missing").status_code == 404
    expert_angles = client.get("/api/eye_lattice?brain=expert").json()
    assert len(expert_angles["left"]) > len(angles["left"])

    sim = live({"notes": []}, 0)
    sim.reset(0.0)
    # Distinct luminance in column-table order; the color channel must not be sent.
    sim.obs = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]])
    image = sim.eye_image()
    assert image is not None
    packet = binary_message(EYE_IMAGE, 1, 0x12345678, image)
    assert struct.unpack_from("<BBHIHH", packet) == (3, 1, 0, 0x12345678, 1, 2)
    assert packet[12:] == bytes([128, 0, 255])


def test_streamed_eye_shows_looming_blur_dropout_and_noise() -> None:
    base = expert_sims()
    live = LiveSaber(
        base.env,
        lambda s, o, c, iv, k: (jnp.array([[12, 4], [12, 4]]), c, jnp.zeros(1)),
        (),
        n_neurons=1,
    )
    note = {
        "time_sec": 1.0,
        "lane": 1,
        "layer": 1,
        "color": 0,
        "cut_direction": 0,
        "is_bomb": False,
    }
    sim = live({"notes": [note]}, 0)

    def pixels(time: float, **controls: float) -> np.ndarray:
        sim.set_interventions({}, controls)
        sim.reset(time)
        packet = sim.eye_image()
        assert packet is not None
        return np.frombuffer(packet, np.uint8, offset=4)

    def footprint(image: np.ndarray) -> int:
        return int((np.abs(image.astype(float) - 127.5) > 3).sum())

    far, near = pixels(0.1), pixels(0.9)
    assert footprint(near) > footprint(far)
    assert footprint(pixels(0.9, blur_deg=10.0)) > footprint(near)
    assert (pixels(0.9, facet_dropout=1.0) == 128).all()
    assert (pixels(0.9, blind_left=1.0, blind_right=1.0, noise=0.1) == 128).all()
    assert not np.array_equal(pixels(0.9, noise=0.1), near)
