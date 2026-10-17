"""Brain dynamics on a tiny hand-built graph."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from brain.config import ModelConfig
from brain.model.brain import (
    Graph,
    eye_inputs,
    init_params,
    init_state,
    make_step,
    param_counts,
    split_heads,
)
from brain.model.interventions import lesion, no_interventions, stimulate

CFG = ModelConfig(
    subgraph_dir="",
    frame_ms=33.333,
    substeps=4,
    tau_init_ms=20.0,
    tau_min_ms=10.0,
    bias_init=0.5,
    activation="relu",
    in_weight_scale=0.9,
    input_scale=1.0,
    action_heads=[3, 2],
)
N = 4


def _graph(sign: float) -> Graph:
    """0 (input) -> 1 -> 2 (DN), with neuron 3 a DN with no inputs."""
    return Graph(
        pre=jnp.array([0, 1], dtype=jnp.int32),
        post=jnp.array([1, 2], dtype=jnp.int32),
        weight=jnp.array([0.9 * sign, 0.9 * sign]),
        pair=jnp.array([0, 1], dtype=jnp.int32),
        dn=jnp.array([2, 3], dtype=jnp.int32),
        input_neuron=jnp.array([0], dtype=jnp.int32),
        input_column=jnp.array([0], dtype=jnp.int32),
        input_channel=jnp.array([0], dtype=jnp.int32),
        input_sign=jnp.array([1.0]),
    )


def _settle(graph: Graph, eye: float, iv=None, frames: int = 60):  # type: ignore[no-untyped-def]
    params = init_params(jax.random.PRNGKey(0), graph, CFG, N)
    step = jax.jit(make_step(graph, CFG, N))
    iv = no_interventions(N) if iv is None else iv
    v = init_state(N)
    for _ in range(frames):
        v, out = step(params, v, jnp.array([[eye], [0.5]]), iv, jax.random.PRNGKey(0))
    return out


def test_shapes_and_finite() -> None:
    out = _settle(_graph(1.0), eye=1.0)
    assert out.logits.shape == (5,) and out.value.shape == () and out.rates.shape == (N,)
    assert np.isfinite(np.asarray(out.rates)).all()


def test_signal_propagates_with_sign() -> None:
    dark, light = _settle(_graph(1.0), 0.0).rates, _settle(_graph(1.0), 1.0).rates
    assert light[2] > dark[2]  # excitatory chain carries light to the DN
    dark, light = _settle(_graph(-1.0), 0.0).rates, _settle(_graph(-1.0), 1.0).rates
    assert light[1] < dark[1]  # inhibitory edge: more input, less output
    assert np.isclose(light[3], dark[3])  # unconnected DN ignores the eye


def test_lesion_silences_and_blocks_output() -> None:
    graph = _graph(1.0)
    iv = lesion(no_interventions(N), jnp.array([1]))
    out = _settle(graph, 1.0, iv)
    assert out.rates[1] == 0.0
    no_edge = graph._replace(weight=jnp.array([0.9, 0.0]))
    assert np.isclose(out.rates[2], _settle(no_edge, 1.0).rates[2])


def test_opto_drives_neuron() -> None:
    iv = stimulate(no_interventions(N), jnp.array([3]), 5.0)
    assert _settle(_graph(1.0), 0.0, iv).rates[3] > _settle(_graph(1.0), 0.0).rates[3] + 4.0


def test_vmap_over_envs() -> None:
    graph = _graph(1.0)
    params = init_params(jax.random.PRNGKey(0), graph, CFG, N)
    step = jax.vmap(make_step(graph, CFG, N), in_axes=(None, 0, 0, None, 0))
    v, out = step(
        params,
        jnp.zeros((8, N)),
        jnp.ones((8, 2, 1)),
        no_interventions(N),
        jax.random.split(jax.random.PRNGKey(0), 8),
    )
    assert v.shape == (8, N) and out.logits.shape == (8, 5)


def test_split_heads_and_counts() -> None:
    heads = split_heads(jnp.arange(5.0), [3, 2])
    assert [h.tolist() for h in heads] == [[0, 1, 2], [3, 4]]
    params = init_params(jax.random.PRNGKey(0), _graph(1.0), CFG, N)
    assert param_counts(params) == {"brain": 3 * N + 2, "readout": 2 * 5 + 5 + 2 + 1, "total": 32}


def test_color_channel_drives_only_its_neurons() -> None:
    graph = _graph(1.0)._replace(input_channel=jnp.array([1], dtype=jnp.int32))
    params = init_params(jax.random.PRNGKey(0), graph, CFG, N)
    step = jax.jit(make_step(graph, CFG, N))
    rates = []
    for color in (0.0, 1.0):
        v = init_state(N)
        for _ in range(60):
            v, out = step(
                params, v, jnp.array([[0.3], [color]]), no_interventions(N), jax.random.PRNGKey(0)
            )
        rates.append(float(out.rates[0]))
    assert rates[1] > rates[0] + 0.5  # neuron 0 now follows channel 1, not luminance


def test_eye_inputs_channels_and_fallback() -> None:
    neurons = pd.DataFrame(
        {
            "type": ["R1-R6", "R7p", "L1", "L1", "L2", "R8y", "T4a"],
            "role": ["columnar"] * 7,
            "side": ["L"] * 7,
            "hex1": [2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0],
            "hex2": [2.0] * 7,
        }
    )
    columns, inputs = eye_inputs(neurons)
    assert len(columns) == 2
    got = sorted(inputs[["neuron", "column", "channel", "sign"]].itertuples(index=False, name=None))
    # Column 0 (2,2) has R1-R6: it takes luminance, R7 takes color, L1 gets nothing.
    # Column 1 (4,2) has only an R8: L1 and L2 take luminance with inverted sign.
    assert got == [(0, 0, 0, 1.0), (1, 0, 1, 1.0), (3, 1, 0, -1.0), (4, 1, 0, -1.0), (5, 1, 1, 1.0)]
