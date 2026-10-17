"""Leaky rate network over the connectome with a DN readout (spec 6).

Owner: Person 1.

    tau_i dv_i/dt = -v_i + sum_j W_ij s_j g_j f(v_j) + b_i + I_i(t) + opto_i
    r_i = gain_i f(v_i)

W_ij comes from synapse counts (log-scaled, normalized per postsynaptic neuron)
and its sparsity is fixed. Trained: per-neuron time constant, bias, and output
scale, plus one scale per (pre cell type, post cell type) pair. Actions and the
value estimate are linear readouts of descending neuron rates only.

Usage:
    graph, columns, n = load_graph(cfg)
    params = init_params(key, graph, cfg, n)
    step = make_step(graph, cfg, n)                 # jit / vmap this
    v, out = step(params, v, eye, interventions, key)
"""

from pathlib import Path
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from brain.config import ModelConfig
from brain.model.interventions import Interventions

# Eye channels (Person 2 contract): 0 luminance, 1 color (red notes high, blue low).
LUMINANCE, COLOR = 0, 1
N_EYE_CHANNELS = 2
LUMINANCE_TYPES = r"R1-R6"
COLOR_TYPES = r"R7.*|R8.*"
FALLBACK_INPUT_TYPES = ("L1", "L2")  # take luminance in columns with no traced R1-R6


class Graph(NamedTuple):
    """Fixed, non-trainable structure. Edges are sorted by `post`."""

    pre: Array  # (E,) int32
    post: Array  # (E,) int32
    weight: Array  # (E,) float32, signed and normalized
    pair: Array  # (E,) int32 index into Params.log_pair
    dn: Array  # (D,) int32 descending neuron indices
    input_neuron: Array  # (M,) int32 neurons that receive eye input
    input_column: Array  # (M,) int32 eye column feeding each input neuron
    input_channel: Array  # (M,) int32 0 luminance (R1-R6, fallback), 1 color (R7, R8)
    input_sign: Array  # (M,) float32 +1 photoreceptor, -1 lamina fallback


class Params(NamedTuple):
    log_tau: Array  # (N,) tau = tau_min + exp(log_tau)
    bias: Array  # (N,)
    log_out: Array  # (N,) synaptic output scale g = exp(log_out), keeps Dale's law
    log_pair: Array  # (P,) per cell-type pair scale
    readout_w: Array  # (D, A)
    readout_b: Array  # (A,)
    value_w: Array  # (D,)
    value_b: Array  # ()


class StepOutput(NamedTuple):
    logits: Array  # (A,) concatenated action heads, see split_heads
    value: Array  # ()
    rates: Array  # (N,) for visualization and analysis


class StepFn(Protocol):
    def __call__(
        self, params: Params, v: Array, eye: Array, iv: Interventions, key: Array
    ) -> tuple[Array, StepOutput]: ...


def load_graph(cfg: ModelConfig) -> tuple[Graph, pd.DataFrame, int]:
    """Build the Graph from subgraph files.

    Also returns the eye column table (column id -> side, hex1, hex2), and the
    neuron count. The eye passes (2, n_columns): luminance and color per column,
    columns in this order.
    """
    d = Path(cfg.subgraph_dir)
    neurons = pd.read_parquet(d / "neurons.parquet")
    edges = pd.read_parquet(d / "edges.parquet")
    edges = edges[edges["sign"] != 0]  # modulatory and unclear: no fast transmission

    w = np.log1p(edges["weight"].to_numpy(dtype=np.float64))
    total_in = np.bincount(edges["post"], weights=w, minlength=len(neurons))
    w = w / total_in[edges["post"]] * cfg.in_weight_scale * edges["sign"].to_numpy()

    types = neurons["type"].to_numpy()
    pair_ids, _ = pd.factorize(pd.Series(types[edges["pre"]]) + ">" + types[edges["post"]])

    columns, inputs = eye_inputs(neurons)
    graph = Graph(
        pre=jnp.asarray(edges["pre"], dtype=jnp.int32),
        post=jnp.asarray(edges["post"], dtype=jnp.int32),
        weight=jnp.asarray(w, dtype=jnp.float32),
        pair=jnp.asarray(pair_ids, dtype=jnp.int32),
        dn=jnp.asarray(np.flatnonzero(neurons["role"] == "descending"), dtype=jnp.int32),
        input_neuron=jnp.asarray(inputs["neuron"], dtype=jnp.int32),
        input_column=jnp.asarray(inputs["column"], dtype=jnp.int32),
        input_channel=jnp.asarray(inputs["channel"], dtype=jnp.int32),
        input_sign=jnp.asarray(inputs["sign"], dtype=jnp.float32),
    )
    return graph, columns, len(neurons)


def eye_inputs(neurons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map eye columns to input neurons (see docs/DATA_NOTES.md, deviation 6).

    Returns the column table and one row per input neuron: neuron index,
    column, channel (0 luminance, 1 color), and sign.
    """
    col = neurons[neurons["role"] == "columnar"].dropna(subset=["hex1", "side"])
    columns = (
        col[["side", "hex1", "hex2"]].drop_duplicates().sort_values(["side", "hex1", "hex2"])
    ).reset_index(drop=True)
    column_id = {tuple(r): i for i, r in enumerate(columns.itertuples(index=False))}
    key = list(zip(col["side"], col["hex1"], col["hex2"], strict=True))
    col = col.assign(column=[column_id[k] for k in key])

    luminance = col[col["type"].str.fullmatch(LUMINANCE_TYPES)]
    color = col[col["type"].str.fullmatch(COLOR_TYPES)]
    fallback = col[
        col["type"].isin(FALLBACK_INPUT_TYPES) & ~col["column"].isin(set(luminance["column"]))
    ]
    inputs = pd.concat(
        [
            luminance.assign(channel=LUMINANCE, sign=1.0),
            color.assign(channel=COLOR, sign=1.0),
            fallback.assign(channel=LUMINANCE, sign=-1.0),
        ]
    )
    inputs = inputs.rename_axis("neuron").reset_index()[["neuron", "column", "channel", "sign"]]
    return columns, inputs


def init_params(key: Array, graph: Graph, cfg: ModelConfig, n_neurons: int) -> Params:
    """Untrained brain: every scale 1, tonic resting bias, small random readout."""
    n_pairs = int(graph.pair.max()) + 1
    n_dn = graph.dn.shape[0]
    n_actions = sum(cfg.action_heads)
    k_read, k_value = jax.random.split(key)
    return Params(
        log_tau=jnp.full(n_neurons, np.log(cfg.tau_init_ms - cfg.tau_min_ms)),
        bias=jnp.full(n_neurons, cfg.bias_init),
        log_out=jnp.zeros(n_neurons),
        log_pair=jnp.zeros(n_pairs),
        readout_w=0.01 * jax.random.normal(k_read, (n_dn, n_actions)),
        readout_b=jnp.zeros(n_actions),
        value_w=0.01 * jax.random.normal(k_value, (n_dn,)),
        value_b=jnp.zeros(()),
    )


def init_state(n_neurons: int) -> Array:
    """Membrane potentials at episode start."""
    return jnp.zeros(n_neurons)


def make_step(graph: Graph, cfg: ModelConfig, n_neurons: int) -> StepFn:
    """One game frame: `substeps` Euler steps of the dynamics, then the readout.

    `eye` is (2, n_columns) in [0, 1]: row 0 luminance, row 1 color, columns ordered
    like load_graph's column table.
    """
    act = {"relu": jax.nn.relu, "softplus": jax.nn.softplus}[cfg.activation]
    dt = cfg.frame_ms / cfg.substeps

    def step(
        params: Params, v: Array, eye: Array, iv: Interventions, key: Array
    ) -> tuple[Array, StepOutput]:
        efficacy = graph.weight * jnp.exp(params.log_pair)[graph.pair]
        out_scale = jnp.exp(params.log_out) * iv.gain
        eye_current = (
            jnp.zeros(n_neurons)
            .at[graph.input_neuron]
            .add(graph.input_sign * eye[graph.input_channel, graph.input_column] * cfg.input_scale)
        )
        drive = eye_current + params.bias + iv.opto
        alpha = dt / (cfg.tau_min_ms + jnp.exp(params.log_tau))

        def substep(v: Array, k: Array) -> tuple[Array, None]:
            r = act(v) * out_scale
            syn = jax.ops.segment_sum(
                efficacy * r[graph.pre], graph.post, n_neurons, indices_are_sorted=True
            )
            noise = iv.noise_std * jax.random.normal(k, (n_neurons,))
            return v + alpha * (-v + syn + drive + noise), None

        v, _ = jax.lax.scan(substep, v, jax.random.split(key, cfg.substeps))
        rates = act(v) * iv.gain
        dn = rates[graph.dn]
        out = StepOutput(
            logits=dn @ params.readout_w + params.readout_b,
            value=dn @ params.value_w + params.value_b,
            rates=rates,
        )
        return v, out

    return step


def split_heads(logits: Array, heads: list[int]) -> list[Array]:
    """Split concatenated logits into one array per action head."""
    return jnp.split(logits, np.cumsum(heads)[:-1].tolist(), axis=-1)


def param_counts(params: Params) -> dict[str, int]:
    """Trainable parameter counts, split into brain and readout (reported in the README)."""
    readout = {"readout_w", "readout_b", "value_w", "value_b"}
    sizes = {k: int(np.size(v)) for k, v in params._asdict().items()}
    brain = sum(s for k, s in sizes.items() if k not in readout)
    head = sum(s for k, s in sizes.items() if k in readout)
    return {"brain": brain, "readout": head, "total": brain + head}
