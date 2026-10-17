"""How much each UI control changes descending neuron (motor) output.

Owner: Person 1. Usage: uv run python scripts/check_controls.py [checkpoint.npz]

Actions read only descending neurons, so a control that leaves them unchanged
cannot change play. Runs a looming stimulus and reports, per control, the mean
relative change in DN rates over the last second and how often the greedy
action of any head changes. Untrained (no checkpoint) the action column only
says the readout is sensitive; the DN change is the meaningful number.
"""

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from brain.config import ModelConfig, load_config
from brain.model.brain import Params, init_params, init_state, load_graph, make_step, split_heads
from brain.model.controls import from_controls, load_controls
from brain.model.interventions import Interventions
from brain.model.stimuli import looming_eye
from training.checkpoints import load_checkpoint

FRAMES = 90


def main(checkpoint: str | None = None) -> None:
    cfg = load_config("configs/model.yaml", ModelConfig)
    graph, columns, n = load_graph(cfg)
    params = init_params(jax.random.PRNGKey(0), graph, cfg, n)
    if checkpoint:
        params, _ = load_checkpoint(checkpoint, params)
    groups = json.loads((Path(cfg.subgraph_dir) / "groups.json").read_text())
    controls = load_controls()
    step = make_step(graph, cfg, n)
    eyes = jnp.asarray(looming_eye(FRAMES, columns))

    @jax.jit
    def run(params: Params, iv: Interventions) -> tuple[jax.Array, jax.Array]:
        def body(v: jax.Array, eye: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            v, out = step(params, v, eye, iv, jax.random.PRNGKey(0))
            return v, (out.rates[graph.dn], out.logits)

        return jax.lax.scan(body, init_state(n), eyes)[1]

    def actions(logits: jax.Array) -> np.ndarray:
        return np.stack(
            [np.asarray(h.argmax(-1)) for h in split_heads(logits, cfg.action_heads)], 1
        )

    base_dn, base_logits = run(params, from_controls({}, groups, controls, n))
    base_dn, base_act = np.asarray(base_dn[-30:]), actions(base_logits[-30:])
    cases: list[tuple[str, dict[str, Any]]] = [
        (f"lesion {s.group}", {"lesion": [s.group]}) for s in controls.sections
    ]
    cases += [
        (f"stimulate {g}", {"stimulate": {g: 5.0}}) for g in ("giant_fiber", "looming_detectors")
    ]
    cases += [
        (f"mood {m.name} {lv:+.0f}", {"mood": {m.name: lv}})
        for m in controls.moods
        for lv in (-1, 1)
    ]
    cases += [("noise 0.5", {"noise": 0.5})]

    print(f"{'control':40s} {'DN change':>10s} {'actions changed':>16s}")
    for name, c in cases:
        dn, logits = run(params, from_controls(c, groups, controls, n))
        dn = np.asarray(dn[-30:])
        rel = np.abs(dn - base_dn).mean() / (np.abs(base_dn).mean() + 1e-9)
        changed = (actions(logits[-30:]) != base_act).any(axis=1).mean()
        print(f"{name:40s} {rel:10.1%} {changed:16.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
