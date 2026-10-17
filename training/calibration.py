"""Fit the brain's readout on real game frames before training.

Owner: Person 1. Two steps, both on synthetic songs played by the expert:

1. Readout normalization (dn_mean, dn_std): measured on game frames, a gray
   background with a few blocks. Flicker input leaves DNs off their game mean.
2. Readout weights: ridge regression from normalized DN rates to the expert's
   action, one-hot per head, then one temperature per head. Gradient descent
   finds this readout very slowly because DN rates are highly correlated: the
   first HiPerGator BC run stayed at the no-vision base rate, while a
   least-squares readout of the same untrained DNs predicts the expert's cell
   on held-out songs at 59% (base rate 27%). BC then fine-tunes everything.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from brain.config import ModelConfig
from brain.model.brain import Graph, Params, make_step, rest_state
from brain.model.interventions import no_interventions
from fly_saber.env import SaberEnv, SaberState
from fly_saber.experts import saber_expert
from fly_saber.maps.adapt import NoteArrays, note_arrays

SONG_SEC = 6.0
NOTE_EVERY_SEC = 0.5
SETTLE_FRAMES = 5  # episodes start from rest_state; skip a few frames anyway
RIDGE = 10.0
TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _random_songs(env: SaberEnv, rng: np.random.Generator, n_songs: int) -> NoteArrays:
    times = np.arange(1.0, SONG_SEC - 0.5, NOTE_EVERY_SEC)[: env.cfg.max_visible_notes]
    songs = [
        note_arrays(
            [
                {
                    "time_sec": float(t),
                    "lane": int(rng.integers(4)),
                    "layer": int(rng.integers(3)),
                    "color": int(rng.integers(2)),
                    "cut_direction": int(rng.integers(4)),
                    "is_bomb": False,
                }
                for t in times
            ],
            env.cfg.max_visible_notes,
        )
        for _ in range(n_songs)
    ]
    stacked: NoteArrays = jax.tree.map(lambda *x: jnp.stack(x), *songs)
    return stacked


def expert_play(
    env: SaberEnv,
    params: Params,
    graph: Graph,
    cfg: ModelConfig,
    n: int,
    key: jax.Array,
    n_songs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw DN rates (songs, frames, D) and expert actions (songs, frames, heads), settled only."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    notes = _random_songs(env, rng, n_songs)
    speed = jnp.full(n_songs, env.cfg.note_speed_mps)
    obs, state = jax.vmap(env.reset)(notes, speed, jnp.zeros(n_songs), jnp.full(n_songs, SONG_SEC))
    step = make_step(graph, dataclasses.replace(cfg, readout_norm=False), n)
    iv = no_interventions(n)

    def body(
        carry: tuple[jax.Array, SaberState, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, SaberState, jax.Array], tuple[jax.Array, jax.Array]]:
        obs, state, v = carry
        action = jax.vmap(lambda s: saber_expert(env, s))(state)
        v, out = jax.vmap(lambda v, o: step(params, v, o, iv, jax.random.PRNGKey(0)))(v, obs)
        stepped = jax.vmap(env.step)(state, action)
        return (stepped[0], stepped[1], v), (out.rates[:, graph.dn], action.reshape(n_songs, -1))

    v0 = jnp.zeros((n_songs, n)) + rest_state(params, graph, cfg, n)
    n_frames = int(SONG_SEC * env.cfg.step_hz)
    dn, actions = jax.jit(lambda c: jax.lax.scan(body, c, None, length=n_frames)[1])(
        (obs, state, v0)
    )
    dn_np = np.swapaxes(np.asarray(dn), 0, 1)[:, SETTLE_FRAMES:]
    actions_np = np.swapaxes(np.asarray(actions), 0, 1)[:, SETTLE_FRAMES:]
    return dn_np, actions_np


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    out: np.ndarray = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    return out


def _best_temperature(scores: np.ndarray, bias: np.ndarray, targets: np.ndarray) -> float:
    """The temperature in TEMPERATURES with the lowest cross-entropy of t * scores + bias."""
    rows = np.arange(len(targets))

    def cross_entropy(t: float) -> float:
        return float(-_log_softmax(t * scores + bias)[rows, targets].mean())

    return min(TEMPERATURES, key=cross_entropy)


def fit_readout(
    env: SaberEnv, params: Params, graph: Graph, cfg: ModelConfig, n: int, key: jax.Array
) -> tuple[Params, dict[str, list[float]]]:
    """Set dn_mean, dn_std, readout_w, readout_b from expert play. Returns params and fit stats.

    Stats per head on a held-out quarter of the songs: accuracy of the fitted readout,
    and of always guessing the most common action.
    """
    dn, actions = expert_play(env, params, graph, cfg, n, key, cfg.calibration_stimuli)
    n_songs, _, d = dn.shape
    mean, std = dn.reshape(-1, d).mean(0), dn.reshape(-1, d).std(0)
    std = np.maximum(std, max(cfg.readout_std_floor * float(np.median(std)), 1e-6))
    z = np.clip((dn - mean) / std, -cfg.readout_clip, cfg.readout_clip) / np.sqrt(d)

    train = np.arange(n_songs) < max(1, n_songs * 3 // 4)
    x_tr, x_te = z[train].reshape(-1, d), z[~train].reshape(-1, d)
    a_tr = actions[train].reshape(len(x_tr), -1)
    a_te = actions[~train].reshape(len(x_te), -1)
    gram = x_tr.T @ x_tr + RIDGE / d * np.eye(d)  # z is already divided by sqrt(d)

    weights, biases = [], []
    stats: dict[str, list[float]] = {"fit_accuracy": [], "base_rate": []}
    for h, size in enumerate(cfg.action_heads):
        y = np.eye(size)[a_tr[:, h]]
        prior = y.mean(0)
        w = np.linalg.solve(gram, x_tr.T @ (y - prior))
        b = np.log(prior + 1e-3)
        weights.append(_best_temperature(x_tr @ w, b, a_tr[:, h]) * w)
        biases.append(b)
        if len(x_te):
            pred = np.argmax(x_te @ weights[-1] + b, axis=1)
            stats["fit_accuracy"].append(round(float(np.mean(pred == a_te[:, h])), 3))
            common = np.bincount(a_tr[:, h], minlength=size).argmax()
            stats["base_rate"].append(round(float(np.mean(a_te[:, h] == common)), 3))

    params = params._replace(
        dn_mean=jnp.asarray(mean, dtype=jnp.float32),
        dn_std=jnp.asarray(std, dtype=jnp.float32),
        readout_w=jnp.asarray(np.concatenate(weights, axis=1), dtype=jnp.float32),
        readout_b=jnp.asarray(np.concatenate(biases), dtype=jnp.float32),
    )
    return params, stats
