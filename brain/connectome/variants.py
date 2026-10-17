"""Rewired and sign-shuffled control brains (spec 5.6).

Owner: Person 1. Each variant is the same neurons with a different edge list, so
every eye column, group, and readout lines up with the real brain.

- rewired: the post-synaptic end of every edge is permuted, which keeps every
  neuron's in-degree and out-degree exactly (a configuration-model shuffle,
  equivalent to many double-edge swaps). Weights and signs stay with the
  presynaptic side, so Dale's law holds. Self-loops and duplicate pairs created
  by the shuffle are swapped away.
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


def rewire(edges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Degree-preserving shuffle of `edges` (columns pre, post, weight, sign)."""
    pre = edges["pre"].to_numpy()
    post = rng.permutation(edges["post"].to_numpy())
    for _ in range(MAX_FIX_ROUNDS):
        bad = np.flatnonzero(_bad(pre, post))
        if len(bad) == 0:
            break
        # Swap each bad edge's target with a random edge's, one pair at a time (vectorized
        # fancy-index swaps lose targets when two bad edges draw the same partner).
        for i, j in zip(bad, rng.integers(0, len(post), len(bad)), strict=True):
            post[i], post[j] = post[j], post[i]
    else:
        raise RuntimeError("rewire did not converge; graph too dense for a simple shuffle")
    out = edges.copy()
    out["post"] = post.astype(edges["post"].dtype)
    return out.sort_values(["post", "pre"], ignore_index=True)


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
