"""Client messages reject invalid controls before reaching the live simulation."""

from typing import Any

import pytest
from pydantic import ValidationError

from server.protocol import client_message_adapter


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
