"""Evaluation conditions and the learning-curve figure."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from brain.model.controls import ControlsConfig, Mood, Section
from scripts.training import plot_learning_curves
from tests.training.test_bc import IDLE_BIAS, bc_config
from tests.training.test_brain_policy import EYE, SABER, model_config, write_subgraph
from training.brain_policy import build
from training.play_eval import Trained, conditions, play_conditions

CONTROLS = ControlsConfig(
    sections=[Section("eye_left", "Left eye", ""), Section("looming_detectors", "Looming", "")],
    moods=[Mood("arousal", "Arousal", "octopamine", 1.0, "Drowsy", "Wired", "")],
    max_stimulation=5.0,
    max_noise=0.5,
)


def test_conditions() -> None:
    assert conditions(CONTROLS, full=False) == {"untouched": {}}
    full = conditions(CONTROLS, full=True)
    assert full["lesion looming_detectors"] == {"lesion": ["looming_detectors"]}
    assert full["arousal -1"] == {"mood": {"arousal": -1.0}}
    assert full["noise max"] == {"noise": 0.5}
    assert list(full)[0] == "untouched" and len(full) == 1 + 2 + 2 + 1


def test_play_conditions_uses_policy_actions_for_every_window(tmp_path: Path) -> None:
    cfg = bc_config(tmp_path)
    subgraph = tmp_path / "subgraph"
    subgraph.mkdir()
    write_subgraph(subgraph)
    env, policy, params = build(
        model_config(subgraph, [13, 5, 13, 5]), SABER, EYE, jax.random.PRNGKey(0)
    )
    params = params._replace(readout_w=jnp.zeros_like(params.readout_w), readout_b=IDLE_BIAS)
    trained = Trained(tmp_path / "test.npz", {}, env, policy, params, {"DN_all": [3]}, 4, cfg)
    results = play_conditions(
        trained,
        CONTROLS,
        {"untouched": {}, "lesioned": {"lesion": ["DN_all"]}},
        "train",
        n_windows=3,
        window_sec=3.0,
    )
    # An idle policy misses every note; an expert-driven rollout would hit them.
    for result in results.values():
        assert result == {"hit_rate": 0.0, "hits": 0, "misses": 18, "bad_cuts": 0, "bombs": 0}


def test_plot_writes_figure_and_table(tmp_path: Path) -> None:
    for variant, run, seed, n in [
        ("real", "fast", 0, 4),
        ("real", "fast", 1, 2),
        ("sign_shuffled", "fast", 0, 4),
    ]:
        d = tmp_path / "ckpt" / variant / run / f"seed{seed}"
        d.mkdir(parents=True)
        rows = [
            {
                "update": 50 * (i + 1),
                "loss": 4.0,
                "hit_rate": 0.2 + 0.01 * i,
                "head_accuracy": [0.6] * 4,
                "bombs": 0,
            }
            for i in range(n)
        ]
        (d / "bc_log.json").write_text(json.dumps(rows))
    runs = plot_learning_curves.load_runs(tmp_path / "ckpt")
    assert [len(v) for v in runs.values()] == [2, 1]
    png = plot_learning_curves.plot(runs, tmp_path / "fig")
    assert png.exists() and (tmp_path / "fig" / "learning_curves.svg").exists()
    table = (tmp_path / "fig" / "learning_curves.md").read_text()
    assert "| Real wiring, brain trained | 1 | 100 | 21.0% |" in table
