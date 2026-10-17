"""Benchmark and sanity-check the brain forward pass (milestone M2).

Owner: Person 1. Usage: uv run python scripts/bench_brain.py [batch ...]

Checks: parameter counts, activity stays bounded, eye input reaches DNs,
gradients through a short rollout are finite, and frames per second.
"""

import sys
import time
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from brain.config import ModelConfig, load_config
from brain.model.brain import (
    Params,
    init_params,
    init_state,
    load_graph,
    make_step,
    param_counts,
)
from brain.model.interventions import no_interventions

FRAMES = 90  # 3 seconds of game time
T = TypeVar("T")


def _wait(x: T) -> T:
    return jax.block_until_ready(x)  # type: ignore[no-untyped-call,no-any-return]


def looming_eye(n_frames: int, columns: pd.DataFrame) -> np.ndarray:
    """A dark disc expanding on a bright background in the left eye, centered on its columns."""
    left = (columns["side"] == "L").to_numpy()
    h = columns[["hex1", "hex2"]].to_numpy(dtype=float)
    center = h[left].mean(axis=0)
    dist = np.linalg.norm(h - center, axis=1)
    radius = np.linspace(0, dist[left].max(), n_frames)[:, None]
    return np.where(left & (dist < radius), 0.0, 1.0).astype(np.float32)


def main(batches: list[int]) -> None:
    cfg = load_config("configs/model.yaml", ModelConfig)
    graph, columns, n = load_graph(cfg)
    print(
        f"neurons {n} | edges {graph.pre.shape[0]} | DNs {graph.dn.shape[0]} | "
        f"eye columns {len(columns)} | input neurons {graph.input_neuron.shape[0]}"
    )
    params = init_params(jax.random.PRNGKey(0), graph, cfg, n)
    print("params:", param_counts(params))

    step = make_step(graph, cfg, n)
    iv = no_interventions(n)

    def rollout(params: Params, eyes: Array, key: Array) -> tuple[Array, Array]:
        def body(v: Array, xs: tuple[Array, Array]) -> tuple[Array, tuple[Array, Array]]:
            eye, k = xs
            v, out = step(params, v, eye, iv, k)
            return v, (out.rates, out.logits)

        keys = jax.random.split(key, eyes.shape[0])
        return jax.lax.scan(body, init_state(n), (eyes, keys))[1]

    run = jax.jit(rollout)
    key = jax.random.PRNGKey(1)
    loom = jnp.asarray(looming_eye(FRAMES, columns))
    gray = jnp.full_like(loom, 1.0)

    rates, _ = run(params, loom, key)
    base, _ = run(params, gray, key)
    rates, base = np.asarray(rates), np.asarray(base)
    roles = pd.read_parquet(f"{cfg.subgraph_dir}/neurons.parquet")["role"].to_numpy()
    print("\nactivity at last frame (looming stimulus):")
    for role in ("columnar", "projection", "intermediate", "descending"):
        r = rates[-1, roles == role]
        print(f"  {role:12s} mean {r.mean():.4f}  max {r.max():.4f}  active {np.mean(r > 0):.0%}")
    print(f"  max rate over all frames: {rates.max():.3f}  (finite: {np.isfinite(rates).all()})")
    dn = np.asarray(graph.dn)
    diff = np.abs(rates[:, dn] - base[:, dn]).mean(axis=1)
    print(
        f"  DN response to looming vs uniform light, mean |diff| by second: "
        f"{[float(f'{x:.2e}') for x in diff.reshape(3, -1).mean(axis=1)]}"
    )

    def loss(params: Params, eyes: Array, key: Array) -> Array:
        _, logits = rollout(params, eyes, key)
        return jnp.mean(logits**2)

    grad_fn = jax.jit(jax.grad(loss))
    g = grad_fn(params, loom[:16], key)
    _wait(g)
    t = time.perf_counter()
    g = _wait(grad_fn(params, loom[:16], key))
    norms = {k: float(jnp.linalg.norm(v)) for k, v in g._asdict().items()}
    print(
        f"\ngrad through 16 frames: {time.perf_counter() - t:.2f}s, "
        f"finite {all(np.isfinite(list(norms.values())))}"
    )
    print("  grad norms:", {k: float(f"{v:.2e}") for k, v in norms.items()})

    print("\nspeed (frames per second of simulated play, all envs combined):")
    for b in batches:
        batched = jax.jit(jax.vmap(rollout, in_axes=(None, None, 0)))
        keys = jax.random.split(key, b)
        _wait(batched(params, loom, keys))
        t = time.perf_counter()
        _wait(batched(params, loom, keys))
        dt = time.perf_counter() - t
        print(
            f"  batch {b:5d}: {b * FRAMES / dt:9.0f} frames/s  "
            f"({dt / FRAMES * 1e3:.2f} ms per frame)"
        )


if __name__ == "__main__":
    main([int(b) for b in sys.argv[1:]] or [1, 16, 64])
