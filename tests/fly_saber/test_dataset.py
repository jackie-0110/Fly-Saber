"""Stacking maps for training and sampling episode windows."""

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from fly_saber.maps.dataset import load_map_set, sample_episodes


def write_map(directory: Path, map_id: str, split: str, times: list[float]) -> None:
    notes = [
        {
            "id": i,
            "time_sec": t,
            "lane": 1,
            "layer": 1,
            "color": 0,
            "cut_direction": 1,
            "is_bomb": False,
        }
        for i, t in enumerate(times)
    ]
    data: dict[str, Any] = {
        "map_id": map_id,
        "split": split,
        "song": {"duration_sec": 30.0},
        "notes": notes,
    }
    (directory / f"{map_id}.json").write_text(json.dumps(data))


def test_load_map_set_stacks_one_split_padded_to_the_longest(tmp_path: Path) -> None:
    write_map(tmp_path, "aaa_easy", "train", [1.0, 2.0])
    write_map(tmp_path, "bbb_hard", "train", [1.0, 1.5, 2.0, 2.5, 3.0])
    write_map(tmp_path, "ccc_hard", "heldout", [1.0])

    map_set, ids = load_map_set(tmp_path, "train", min_notes=4)

    assert ids == ["aaa_easy", "bbb_hard"]
    assert map_set.notes.time_sec.shape == (2, 5)
    assert np.asarray(map_set.notes.valid).sum(axis=1).tolist() == [2, 5]


def test_load_map_set_thins_close_notes(tmp_path: Path) -> None:
    write_map(tmp_path, "aaa_easy", "train", [1.0, 1.2, 2.0])

    map_set, _ = load_map_set(tmp_path, "train", min_notes=1, min_gap_sec=0.5)

    assert np.asarray(map_set.notes.time_sec[0]).tolist() == [1.0, 2.0]


def test_sample_episodes_stays_inside_each_song(tmp_path: Path) -> None:
    write_map(tmp_path, "aaa_easy", "train", [1.0])
    map_set, _ = load_map_set(tmp_path, "train", min_notes=1)

    episodes = sample_episodes(jax.random.PRNGKey(0), map_set, n=64, window_sec=10.0, speed_mps=7.0)

    assert episodes.notes.time_sec.shape == (64, 1)
    assert (np.asarray(episodes.start_sec) >= 0.0).all()
    assert (np.asarray(episodes.end_sec) <= 30.0).all()
    assert (np.asarray(episodes.speed_mps) == 7.0).all()


def test_load_map_set_keeps_maps_up_to_a_difficulty_quantile(tmp_path: Path) -> None:
    write_map(tmp_path, "aaa_easy", "train", [1.0, 3.0, 5.0])
    write_map(tmp_path, "bbb_hard", "train", [1.0, 1.2, 1.4, 1.6, 1.8])

    _, ids = load_map_set(tmp_path, "train", min_notes=1, difficulty_quantile=0.5)

    assert ids == ["aaa_easy"]
