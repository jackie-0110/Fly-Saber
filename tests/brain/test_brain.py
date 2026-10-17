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
N = 5  # 0 input -> 1 -> 2 (DN); 3 DN with no inputs; 4 modulatory, no inputs


def _graph(sign: float, mod_gain: float | None = None) -> Graph:
    """0 (input) -> 1 -> 2 (DN), with neuron 3 a DN with no inputs.

    With `mod_gain`, modulatory neuron 4 modulates DN 2 with that log-gain per unit rate.
    """
    mod = mod_gain is not None
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
        mod_pre=jnp.array([4] if mod else [], dtype=jnp.int32),
        mod_post=jnp.array([2] if mod else [], dtype=jnp.int32),
        mod_weight=jnp.array([mod_gain] if mod else [], dtype=jnp.float32),
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


def test_modulation_is_neutral_at_rest() -> None:
    plain = _settle(_graph(1.0), 1.0).rates
    modulated = _settle(_graph(1.0, mod_gain=1.0), 1.0).rates
    np.testing.assert_allclose(modulated, plain, rtol=1e-5)


def test_modulator_scales_its_target() -> None:
    graph, base = _graph(1.0, mod_gain=1.0), _settle(_graph(1.0), 1.0).rates
    up = _settle(graph, 1.0, stimulate(no_interventions(N), jnp.array([4]), 1.0)).rates
    down = _settle(graph, 1.0, lesion(no_interventions(N), jnp.array([4]))).rates
    # Rest rate 0.5. Driving the modulator +1 multiplies DN 2 by e; silencing it by e^-0.5.
    assert np.isclose(up[2] / base[2], np.e, rtol=1e-3)
    assert np.isclose(down[2] / base[2], np.exp(-0.5), rtol=1e-3)
    assert np.isclose(up[3], base[3])  # neurons it does not reach are untouched
    inhibitory = _settle(
        _graph(1.0, mod_gain=-1.0), 1.0, stimulate(no_interventions(N), jnp.array([4]), 1.0)
    )
    assert inhibitory.rates[2] < base[2]


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
    assert param_counts(params) == {
        "brain": 3 * N + 2,
        "readout": 2 * 5 + 5 + 2 + 1,
        "total": 3 * N + 2 + 18,
    }


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


def test_relu_sat_levels_off() -> None:
    import dataclasses

    cfg = dataclasses.replace(CFG, activation="relu_sat", rate_max=2.0)
    graph = _graph(1.0)
    params = init_params(jax.random.PRNGKey(0), graph, cfg, N)
    step = jax.jit(make_step(graph, cfg, N))
    v = init_state(N)
    iv = stimulate(no_interventions(N), jnp.array([3]), 100.0)
    for _ in range(60):
        v, out = step(params, v, jnp.array([[1.0], [0.5]]), iv, jax.random.PRNGKey(0))
    assert 1.99 < float(out.rates[3]) <= 2.0  # huge drive, capped at rate_max


def test_remat_gives_identical_gradients() -> None:
    import dataclasses

    graph = _graph(1.0, mod_gain=1.0)
    params = init_params(jax.random.PRNGKey(0), graph, CFG, N)

    def loss(remat: bool):  # type: ignore[no-untyped-def]
        step = make_step(graph, dataclasses.replace(CFG, remat=remat), N)

        def f(p):  # type: ignore[no-untyped-def]
            v, total = init_state(N), jnp.zeros(())
            for _ in range(5):
                v, out = step(
                    p, v, jnp.array([[1.0], [0.5]]), no_interventions(N), jax.random.PRNGKey(0)
                )
                total = total + jnp.sum(out.logits**2)
            return total

        return jax.grad(f)(params)

    for a, b in zip(jax.tree.leaves(loss(True)), jax.tree.leaves(loss(False)), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_readout_norm_calibrates_and_is_never_trained() -> None:
    import dataclasses

    graph = _graph(1.0)
    raw = init_params(jax.random.PRNGKey(0), graph, dataclasses.replace(CFG, readout_norm=False), N)
    assert np.all(np.asarray(raw.dn_std) == 1) and np.all(np.asarray(raw.dn_mean) == 0)

    params = init_params(jax.random.PRNGKey(0), graph, CFG, N)
    # DN 2 follows the flickering eye, DN 3 has no input: it gets the std floor, not ~0.
    assert float(params.dn_mean[0]) > 0 and float(params.dn_std[0]) > float(params.dn_std[1])
    # Floor is 0.1 x the median of the raw stds, (std0 + 0) / 2.
    assert np.isclose(float(params.dn_std[1]), 0.05 * float(params.dn_std[0]), rtol=1e-3)

    step = make_step(graph, CFG, N)

    def loss(p):  # type: ignore[no-untyped-def]
        _, out = step(
            p, init_state(N), jnp.array([[1.0], [0.5]]), no_interventions(N), jax.random.PRNGKey(0)
        )
        return jnp.sum(out.logits**2) + out.value**2

    g = jax.grad(loss)(params)
    # dn_std never gets gradients. dn_mean does (the live baseline needs them); optimizers
    # keep the stored leaf fixed (tests/training/test_bc.py).
    assert float(jnp.abs(g.dn_std).max()) == 0.0
    assert float(jnp.abs(g.readout_w).max()) > 0.0


def test_freeze_brain_stops_gradients_at_dns() -> None:
    import dataclasses

    graph = _graph(1.0)
    cfg = dataclasses.replace(CFG, freeze_brain=True)
    params = init_params(jax.random.PRNGKey(0), graph, cfg, N)
    step = make_step(graph, cfg, N)

    def loss(p):  # type: ignore[no-untyped-def]
        _, out = step(
            p, init_state(N), jnp.array([[1.0], [0.5]]), no_interventions(N), jax.random.PRNGKey(0)
        )
        return jnp.sum(out.logits**2)

    g = jax.grad(loss)(params)
    assert float(jnp.abs(g.bias).max()) == 0.0 and float(jnp.abs(g.log_pair).max()) == 0.0
    assert float(jnp.abs(g.readout_w).max()) > 0.0
