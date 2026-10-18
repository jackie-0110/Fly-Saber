"""Rewired and sign-shuffled control brains (spec 5.6).

Owner: Person 1. Each variant is the same neurons with a different edge list, so
every eye column, group, and readout lines up with the real brain.

- rewired: the post-synaptic end of every edge is permuted, which keeps every
  neuron's in-degree and out-degree exactly (a configuration-model shuffle,
  equivalent to many double-edge swaps). Weights and signs stay with the
  presynaptic side, so Dale's law holds. Self-loops and duplicate pairs created
  by the shuffle are swapped away.
- rewired_layered: the same shuffle, but only within each (pre layer, post layer)
  block (photoreceptors, lamina, medulla, T4/T5, projection, central relay,
  descending, modulatory). The fly's layered pathway survives: plain rewiring
  wires the eyes straight into the DNs (1,068 direct eye-to-DN edges versus 1),
  which makes vision easier to read out and is not a fair control.
- sign_shuffled: real wiring, neuron signs (and so transmitter labels) permuted.
"""

import numpy as np
import pandas as pd

MAX_FIX_ROUNDS = 200


def _bad(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Edges that are self-loops or repeat an earlier (pre, post) pair."""
    pairs = pd.DataFrame({"pre": pre, "post": post})
    bad: np.ndarray = (pre == post) | pairs.duplicated().to_numpy()
    return bad


def _shuffle_targets(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute `post`, then swap away self-loops and duplicate pairs. Raises if it cannot."""
    post = rng.permutation(post)
    for _ in range(MAX_FIX_ROUNDS):
        bad = np.flatnonzero(_bad(pre, post))
        if len(bad) == 0:
            return post
        # Swap each bad edge's target with a random edge's, one pair at a time (vectorized
        # fancy-index swaps lose targets when two bad edges draw the same partner).
        for i, j in zip(bad, rng.integers(0, len(post), len(bad)), strict=True):
            post[i], post[j] = post[j], post[i]
    raise RuntimeError("shuffle did not converge; block too dense for a simple shuffle")


def rewire(edges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Degree-preserving shuffle of `edges` (columns pre, post, weight, sign)."""
    out = edges.copy()
    out["post"] = _shuffle_targets(edges["pre"].to_numpy(), edges["post"].to_numpy(), rng)
    out["post"] = out["post"].astype(edges["post"].dtype)
    return out.sort_values(["post", "pre"], ignore_index=True)


def neuron_layers(neurons: pd.DataFrame) -> pd.Series:
    """Layer of each neuron along the visual-to-motor pathway (index = neuron index)."""
    t, role = neurons["type"], neurons["role"]
    layer = role.copy()
    layer[t.str.fullmatch(r"R1-R6|R7.*|R8.*")] = "photoreceptor"
    layer[t.str.fullmatch(r"L[1-5]")] = "lamina"
    layer[(role == "columnar") & t.str.fullmatch(r"Mi\d+|Tm\d+.*")] = "medulla"
    layer[t.str.fullmatch(r"T4[a-d]|T5[a-d]")] = "T4T5"
    return layer


def rewire_layered(
    edges: pd.DataFrame, neurons: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Shuffle targets within each (pre layer, post layer) block.

    Every neuron keeps its in- and out-degree within each block, so the layer-to-layer
    connection counts are exactly the real ones. Blocks too dense to shuffle without
    duplicate pairs stay as they are; their edge count is returned.
    """
    layer = neuron_layers(neurons).to_numpy()
    out = edges.copy()
    blocks = pd.Series(layer[edges["pre"]]) + ">" + layer[edges["post"]]
    stats = {"blocks": 0, "kept_edges": 0}
    for _, idx in blocks.groupby(blocks).groups.items():
        rows = np.asarray(idx)
        stats["blocks"] += 1
        try:
            out.loc[rows, "post"] = _shuffle_targets(
                edges["pre"].to_numpy()[rows], edges["post"].to_numpy()[rows], rng
            )
        except RuntimeError:
            stats["kept_edges"] += len(rows)
    out["post"] = out["post"].astype(edges["post"].dtype)
    return out.sort_values(["post", "pre"], ignore_index=True), stats


def shuffle_signs(
    neurons: pd.DataFrame, edges: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute neuron signs and transmitters together; edges take their new pre sign."""
    order = rng.permutation(len(neurons))
    out_n = neurons.copy()
    out_n["sign"] = neurons["sign"].to_numpy()[order]
    out_n["nt"] = neurons["nt"].to_numpy()[order]
    out_e = edges.copy()
    out_e["sign"] = out_n["sign"].to_numpy()[edges["pre"].to_numpy()]
    return out_n, out_e
