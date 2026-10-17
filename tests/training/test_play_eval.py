"""Evaluation conditions and the learning-curve figure."""

import json
import sys
from pathlib import Path

from brain.model.controls import ControlsConfig, Mood, Section
from training.play_eval import conditions

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import plot_learning_curves  # noqa: E402

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
