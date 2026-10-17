"""UI controls to interventions."""

import numpy as np
import pytest

from brain.model.controls import ControlsConfig, Mood, Section, from_controls, load_controls

GROUPS = {"eye_left": [0, 1], "octopamine": [2], "giant_fiber": [3]}
CFG = ControlsConfig(
    sections=[Section("eye_left", "Left eye", "")],
    moods=[Mood("arousal", "Arousal", "octopamine", 1.0, "Drowsy", "Wired", "")],
    max_stimulation=5.0,
    max_noise=0.5,
)


def test_empty_is_unmodified() -> None:
    iv = from_controls({}, GROUPS, CFG, 5)
    assert np.all(np.asarray(iv.gain) == 1) and np.all(np.asarray(iv.opto) == 0)
    assert float(iv.noise_std) == 0.0


def test_lesion_stimulate_mood_noise() -> None:
    iv = from_controls(
        {
            "lesion": ["eye_left"],
            "stimulate": {"giant_fiber": 9.0},
            "mood": {"arousal": -2},
            "noise": 0.2,
        },
        GROUPS,
        CFG,
        5,
    )
    assert np.asarray(iv.gain).tolist() == [0, 0, 1, 1, 1]
    assert np.asarray(iv.opto).tolist() == [0, 0, -1.0, 5.0, 0]  # clamped to max and to -1..1
    assert np.isclose(float(iv.noise_std), 0.2)


def test_lesion_overrides_stimulation() -> None:
    iv = from_controls({"lesion": ["giant_fiber"], "stimulate": {"giant_fiber": 5}}, GROUPS, CFG, 5)
    assert float(iv.gain[3]) == 0.0 and float(iv.opto[3]) == 0.0


@pytest.mark.parametrize(
    "controls",
    [{"lesion": ["nope"]}, {"mood": {"nope": 1}}, {"stimulate": {"eye_left": "a"}}, {"typo": []}],
)
def test_rejects_bad_input(controls: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        from_controls(controls, GROUPS, CFG, 5)


def test_shipped_controls_reference_real_groups() -> None:
    import json
    from pathlib import Path

    cfg = load_controls()
    path = Path("data/processed/malecns_subgraph/groups.json")
    if not path.exists():
        pytest.skip("subgraph not built")
    groups = json.loads(path.read_text())
    names = [s.group for s in cfg.sections] + [m.group for m in cfg.moods]
    assert [g for g in names if g not in groups] == []
