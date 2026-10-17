"""Adapting normalized maps for the fly."""

from typing import Any

import numpy as np
import pytest

from fly_saber.maps.adapt import difficulty, note_arrays, thin_notes


def note(
    time_sec: float, color: int | None = 0, cut: int | None = 0, lane: int = 0
) -> dict[str, Any]:
    return {
        "time_sec": time_sec,
        "lane": lane,
        "layer": 0,
        "color": color,
        "cut_direction": cut,
        "is_bomb": color is None,
    }


def test_diagonals_keep_vertical_component_and_any_maps_to_any() -> None:
    cuts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    arrays = note_arrays([note(t, cut=c) for t, c in enumerate(cuts)], max_notes=9)
    np.testing.assert_array_equal(arrays.direction, [0, 1, 2, 3, 0, 0, 1, 1, 4])


def test_padding_rows_are_invalid_and_bombs_have_bomb_color() -> None:
    arrays = note_arrays([note(0.0), note(1.0, color=None, cut=None)], max_notes=4)
    np.testing.assert_array_equal(arrays.valid, [True, True, False, False])
    np.testing.assert_array_equal(arrays.color[:2], [0, -1])


def test_too_many_notes_raise() -> None:
    with pytest.raises(ValueError):
        note_arrays([note(0.0), note(1.0)], max_notes=1)


def test_thinning_is_per_color_and_keeps_bombs() -> None:
    notes = [note(0.0, 0), note(0.1, 1), note(0.2, 0), note(0.25, None, None), note(0.6, 0)]
    kept = thin_notes(notes, min_gap_sec=0.5)
    assert [n["time_sec"] for n in kept] == [0.0, 0.1, 0.25, 0.6]


def test_denser_changing_maps_score_harder() -> None:
    calm = [note(t, cut=0) for t in range(5)]
    busy = [note(t / 2, cut=t % 2, lane=t % 4) for t in range(10)]
    assert difficulty(calm)["score"] < difficulty(busy)["score"]


def test_difficulty_is_finite_when_no_saber_has_two_notes() -> None:
    assert difficulty([note(0.0, 0), note(1.0, 1)])["score"] == 2.0
