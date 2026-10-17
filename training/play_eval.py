"""Score trained brains by letting them play: held-out songs, lesions, moods.

Owner: Person 1. Used by scripts/eval_checkpoints.py.

Greedy play only (the expert never drives). Every brain and every condition
plays the same song windows (fixed key), so differences are paired.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from brain.config import ModelConfig
from brain.model.brain import (
    Params,
    init_params,
    load_graph,
    make_step,
    rest_state,
    with_live_baseline,
)
from brain.model.controls import ControlsConfig, from_controls
from brain.model.interventions import Interventions, no_interventions
from fly_saber.env import SaberConfig, SaberEnv
from fly_saber.eye import EyeConfig
from fly_saber.maps.dataset import load_map_set, sample_episodes
from training.bc import BCConfig, rollout
from training.brain_policy import BrainPolicy, column_env
from training.checkpoints import load_checkpoint, read_meta, subgraph_fingerprint


@dataclass(frozen=True)
class Trained:
    """A checkpoint ready to play."""

    path: Path
    meta: dict[str, Any]
    env: SaberEnv
    policy: BrainPolicy
    params: Params
    groups: dict[str, list[int]]
    n_neurons: int
    train_cfg: BCConfig


def _rebuild(cls: type[Any], values: dict[str, Any]) -> Any:
    """A config dataclass from saved values, ignoring fields this code no longer has."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


def load_trained(path: str | Path, subgraph_dir: str | Path | None = None) -> Trained:
    """Load a BC checkpoint with the configs it was trained with.

    Uses the subgraph files at `subgraph_dir` (default: the saved one, relative to the
    working directory) and refuses to load if they do not match the checkpoint's
    fingerprint. For a bundle from scripts/bundle_checkpoint.py, pass its subgraph/.
    """
    path = Path(path)
    meta = read_meta(path)
    configs = meta["configs"]
    model_cfg: ModelConfig = _rebuild(ModelConfig, configs["model"])
    if subgraph_dir is not None:
        model_cfg = dataclasses.replace(model_cfg, subgraph_dir=str(subgraph_dir))
    saber_cfg: SaberConfig = _rebuild(SaberConfig, configs["saber"])
    eye_cfg: EyeConfig = _rebuild(EyeConfig, configs["eye"])
    train_cfg: BCConfig = _rebuild(BCConfig, configs["train"])

    graph, columns, n = load_graph(model_cfg)
    template_cfg = dataclasses.replace(model_cfg, readout_norm=False)  # shapes only, no fit
    template = init_params(jax.random.PRNGKey(0), graph, template_cfg, n)
    params, _ = load_checkpoint(
        path, template, expect_subgraph=subgraph_fingerprint(model_cfg.subgraph_dir)
    )
    rest = rest_state(params, graph, model_cfg, n)
    if model_cfg.readout_norm and model_cfg.live_baseline:
        params = with_live_baseline(params, graph, model_cfg, n, rest)
    env = column_env(saber_cfg, eye_cfg, columns)
    policy = BrainPolicy(make_step(graph, model_cfg, n), n, no_interventions(n), rest)
    groups = json.loads((Path(model_cfg.subgraph_dir) / "groups.json").read_text())
    return Trained(path, meta, env, policy, params, groups, n, train_cfg)


def conditions(controls: ControlsConfig, full: bool) -> dict[str, dict[str, Any]]:
    """Named UI-control settings to evaluate. `full=False` gives only the untouched brain."""
    out: dict[str, dict[str, Any]] = {"untouched": {}}
    if not full:
        return out
    for s in controls.sections:
        out[f"lesion {s.group}"] = {"lesion": [s.group]}
    for m in controls.moods:
        out[f"{m.name} -1"] = {"mood": {m.name: -1.0}}
        out[f"{m.name} +1"] = {"mood": {m.name: 1.0}}
    out["noise max"] = {"noise": controls.max_noise}
    return out


def play_conditions(
    trained: Trained,
    controls: ControlsConfig,
    named: dict[str, dict[str, Any]],
    split: str,
    n_windows: int,
    window_sec: float,
    seed: int = 1234,
) -> dict[str, dict[str, float]]:
    """Hit rate and counts per condition on `split` songs (same windows for every condition)."""
    cfg = trained.train_cfg
    env = trained.env
    map_set, _ = load_map_set(Path(cfg.maps_dir), split, env.cfg.max_visible_notes, cfg.min_gap_sec)
    k_ep, k_play = jax.random.split(jax.random.PRNGKey(seed))
    episodes = sample_episodes(k_ep, map_set, n_windows, window_sec, env.cfg.note_speed_mps)
    n_steps = math.ceil(window_sec * env.cfg.step_hz) + 1

    @jax.jit
    def play(iv: Interventions) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        policy = dataclasses.replace(trained.policy, interventions=iv)
        drives = jnp.ones(n_windows, bool)
        m = rollout(env, policy, trained.params, episodes, k_play, n_steps, drives)
        return m.hits, m.misses, m.bad_cuts, m.bombs

    results = {}
    for name, ui in named.items():
        iv = from_controls(ui, trained.groups, controls, trained.n_neurons)
        hits, misses, bad, bombs = (int(x) for x in play(iv))
        judged = hits + misses + bad
        results[name] = {
            "hit_rate": hits / judged if judged else 0.0,
            "hits": hits,
            "misses": misses,
            "bad_cuts": bad,
            "bombs": bombs,
        }
    return results
