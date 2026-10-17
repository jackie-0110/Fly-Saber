"""Behavior cloning from the scripted expert (spec 12.1).

Owner: Person 2. See FLYARCADE_HANDOFF.md for the spec.

Any recurrent policy trains here: the connectome brain, the MLP baseline, or a
test stub. The env is driven by the expert's actions while the policy is scored
against them, so every update sees fresh episodes and no dataset is stored.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax

from brain.config import ModelConfig
from brain.config import load_config as load_model_config
from fly_saber.config import CONFIG_DIR, load_config
from fly_saber.env import IDLE_CELL, NO_CUT, SaberConfig, SaberEnv
from fly_saber.experts import saber_expert
from fly_saber.eye import EyeConfig
from fly_saber.maps.dataset import Episodes, MapSet, load_map_set, sample_episodes
from training.brain_policy import build
from training.checkpoints import save_checkpoint, subgraph_fingerprint

# Per saber: position head (12 cells + idle), then direction head (4 cuts + no cut).
HEADS_PER_SABER = (IDLE_CELL + 1, NO_CUT + 1)


@dataclass(frozen=True)
class BCConfig:
    maps_dir: str
    split: str
    min_gap_sec: float
    window_sec: float
    speed_mps: float
    batch_size: int
    learning_rate: float
    n_updates: int
    log_every: int
    seed: int
    variant: str
    checkpoint_dir: str


class Policy(Protocol):
    """One env's policy; `bc` vmaps it. `carry` is the recurrent state (brain voltages)."""

    def init_carry(self) -> Any: ...

    def apply(
        self, params: Any, carry: Any, obs: jax.Array, key: jax.Array
    ) -> tuple[Any, jax.Array]:
        """Return the next carry and concatenated logits, heads ordered as `action_heads`."""
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Metrics:
    loss: jax.Array
    """Mean summed cross-entropy over the heads, per live env step."""
    head_accuracy: jax.Array
    """(n_heads,) fraction of live steps where the argmax matches the expert."""
    hits: jax.Array
    misses: jax.Array
    bad_cuts: jax.Array
    bombs: jax.Array


def action_heads(n_sabers: int) -> list[int]:
    return list(HEADS_PER_SABER) * n_sabers


def split_heads(logits: jax.Array, heads: list[int]) -> list[jax.Array]:
    return jnp.split(logits, np.cumsum(heads)[:-1].tolist())


def cross_entropy(
    logits: jax.Array, action: jax.Array, heads: list[int]
) -> tuple[jax.Array, jax.Array]:
    """Summed cross-entropy against an (n_sabers, 2) action, and per-head argmax matches."""
    targets = action.reshape(-1)
    parts = split_heads(logits, heads)
    nll = sum(-jax.nn.log_softmax(p)[targets[i]] for i, p in enumerate(parts))
    correct = jnp.stack([jnp.argmax(p) == targets[i] for i, p in enumerate(parts)])
    return jnp.asarray(nll), correct


def greedy_action(logits: jax.Array, heads: list[int]) -> jax.Array:
    return jnp.stack([jnp.argmax(p) for p in split_heads(logits, heads)]).reshape(-1, 2)


def rollout(
    env: SaberEnv,
    policy: Policy,
    params: Any,
    episodes: Episodes,
    key: jax.Array,
    n_steps: int,
    drive_with_expert: bool,
) -> Metrics:
    """Run a batch of episodes, scoring the policy against the expert at every step.

    With `drive_with_expert` the expert's actions move the env (training); otherwise
    the policy's greedy actions do, and the hit counts measure how well it plays.
    """
    heads = action_heads(env.cfg.n_sabers)
    n_envs = episodes.speed_mps.shape[0]
    obs, state = jax.vmap(env.reset)(
        episodes.notes, episodes.speed_mps, episodes.start_sec, episodes.end_sec
    )
    carry = jax.vmap(lambda _: policy.init_carry())(jnp.arange(n_envs))

    def body(loop: tuple[Any, ...], key: jax.Array) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        obs, state, carry, alive = loop
        apply = jax.vmap(policy.apply, in_axes=(None, 0, 0, 0))
        carry, logits = apply(params, carry, obs, jax.random.split(key, n_envs))
        expert = jax.vmap(lambda s: saber_expert(env, s))(state)
        nll, correct = jax.vmap(lambda lg, a: cross_entropy(lg, a, heads))(logits, expert)
        action = (
            expert if drive_with_expert else jax.vmap(lambda lg: greedy_action(lg, heads))(logits)
        )
        obs, state, _, done, _ = jax.vmap(env.step)(state, action)
        live = alive.astype(jnp.float32)
        return (obs, state, carry, alive & ~done), (nll * live, correct * live[:, None], live)

    init = (obs, state, carry, jnp.ones(n_envs, bool))
    (_, state, _, _), (nll, correct, live) = jax.lax.scan(
        body, init, jax.random.split(key, n_steps)
    )
    n_live = jnp.maximum(live.sum(), 1.0)
    return Metrics(
        loss=nll.sum() / n_live,
        head_accuracy=correct.sum(axis=(0, 1)) / n_live,
        hits=state.hits.sum(),
        misses=state.misses.sum(),
        bad_cuts=state.bad_cuts.sum(),
        bombs=state.bombs.sum(),
    )


def make_steps(
    env: SaberEnv,
    policy: Policy,
    map_set: MapSet,
    cfg: BCConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[Callable[..., Any], Callable[..., Metrics]]:
    """Jitted (update, evaluate) over random windows of `map_set`."""
    # reset lets an episode run two timing windows past end_sec to judge its last notes.
    n_steps = math.ceil((cfg.window_sec + 2 * env.cfg.timing_window_sec) * env.cfg.step_hz) + 1

    def episodes_and_key(key: jax.Array) -> tuple[Episodes, jax.Array]:
        k_episodes, k_rollout = jax.random.split(key)
        return sample_episodes(
            k_episodes, map_set, cfg.batch_size, cfg.window_sec, cfg.speed_mps
        ), k_rollout

    def loss(params: Any, key: jax.Array) -> tuple[jax.Array, Metrics]:
        episodes, key = episodes_and_key(key)
        metrics = rollout(env, policy, params, episodes, key, n_steps, drive_with_expert=True)
        return metrics.loss, metrics

    @jax.jit
    def update(params: Any, opt_state: Any, key: jax.Array) -> tuple[Any, Any, Metrics]:
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    @jax.jit
    def evaluate(params: Any, key: jax.Array) -> Metrics:
        episodes, key = episodes_and_key(key)
        return rollout(env, policy, params, episodes, key, n_steps, drive_with_expert=False)

    return update, evaluate


def train(
    env: SaberEnv, policy: Policy, params: Any, cfg: BCConfig
) -> tuple[Any, list[dict[str, Any]]]:
    """Clone the expert into `params`; returns them and one log row per `log_every` updates."""
    map_set, _ = load_map_set(
        Path(cfg.maps_dir), cfg.split, env.cfg.max_visible_notes, cfg.min_gap_sec
    )
    optimizer = optax.adam(cfg.learning_rate)
    update, evaluate = make_steps(env, policy, map_set, cfg, optimizer)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(cfg.seed)
    log = []
    for i in range(1, cfg.n_updates + 1):
        key, k_update, k_eval = jax.random.split(key, 3)
        params, opt_state, metrics = update(params, opt_state, k_update)
        if i % cfg.log_every == 0 or i == cfg.n_updates:
            played = evaluate(params, k_eval)
            judged = int(played.hits + played.misses + played.bad_cuts)
            row = {
                "update": i,
                "loss": float(metrics.loss),
                "head_accuracy": [round(float(a), 4) for a in metrics.head_accuracy],
                "hit_rate": int(played.hits) / judged if judged else 0.0,
                "bombs": int(played.bombs),
            }
            log.append(row)
            print(row, flush=True)
    return params, log


def run(cfg: BCConfig, model_cfg: ModelConfig, saber_cfg: SaberConfig, eye_cfg: EyeConfig) -> Path:
    """Clone the expert into a fresh brain and save it; returns the checkpoint path."""
    env, policy, params = build(model_cfg, saber_cfg, eye_cfg, jax.random.PRNGKey(cfg.seed))
    params, log = train(env, policy, params, cfg)
    run_dir = Path(cfg.checkpoint_dir) / cfg.variant / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bc_log.json").write_text(json.dumps(log, indent=1))
    return save_checkpoint(
        run_dir / f"bc_{cfg.n_updates}.npz",
        params,
        variant=cfg.variant,
        stage="bc",
        step=cfg.n_updates,
        configs={"model": model_cfg, "saber": saber_cfg, "eye": eye_cfg, "train": cfg},
        subgraph=subgraph_fingerprint(model_cfg.subgraph_dir),
        metrics=log[-1],
    )


def main() -> None:
    path = run(
        load_config(BCConfig, "train_bc"),
        load_model_config(CONFIG_DIR / "model.yaml", ModelConfig),
        load_config(SaberConfig, "saber"),
        load_config(EyeConfig, "eye"),
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
