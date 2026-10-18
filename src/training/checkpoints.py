"""Checkpoint save and load with config hash (spec 12.4).

Owner: Person 1. Format is documented in docs/reference/PROTOCOL.md section 5.

A checkpoint is one .npz file: one array per parameter leaf, keyed by its
pytree path (for example ".log_tau"), plus a JSON "__meta__" entry. Loading
needs a template pytree of the same structure (from `init_params`), so any
policy's params work: the connectome brain, the MLP baseline, or a stub.
"""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import jax
import numpy as np
import pandas as pd

FORMAT_VERSION = 1
META_KEY = "__meta__"
T = TypeVar("T")


def _plain(config: Any) -> Any:
    return asdict(config) if is_dataclass(config) and not isinstance(config, type) else config


def config_hash(configs: dict[str, Any]) -> str:
    """Short stable hash of named configs (dataclasses or plain dicts)."""
    canonical = json.dumps({k: _plain(v) for k, v in configs.items()}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def subgraph_fingerprint(subgraph_dir: str | Path) -> dict[str, Any]:
    """Identity of the connectome a checkpoint was trained on.

    Counts from meta.json, plus content hashes of the neuron order (body ids) and the
    edge list, so a rebuilt subgraph with the same counts but different neurons in
    different slots is caught.
    """
    d = Path(subgraph_dir)
    meta = json.loads((d / "meta.json").read_text())
    fingerprint: dict[str, Any] = {k: meta[k] for k in ("n_neurons", "n_edges", "n_synapses")}
    if meta.get("variant", "real") != "real":  # controls have the same counts
        fingerprint["variant"] = meta["variant"]
    neurons = pd.read_parquet(d / "neurons.parquet")
    if "body_id" in neurons:  # hand-built test subgraphs may not have body ids
        fingerprint["neurons_sha256"] = _sha(neurons["body_id"].to_numpy(dtype=np.int64))
    edges = pd.read_parquet(d / "edges.parquet", columns=["pre", "post", "weight"])
    fingerprint["edges_sha256"] = _sha(edges.to_numpy(dtype=np.int64))
    return fingerprint


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def subgraph_mismatch(trained: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Fingerprint keys that differ. Only keys the checkpoint recorded are compared, so
    older checkpoints (counts only) still load; newer ones are checked by content."""
    return [k for k in trained if trained[k] != current.get(k)]


def save_checkpoint(
    path: str | Path,
    params: Any,
    *,
    variant: str,
    stage: str,
    step: int,
    configs: dict[str, Any],
    subgraph: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write `params` and metadata to `path` (.npz). Returns the path written."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves}
    meta = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "stage": stage,
        "step": step,
        "config_hash": config_hash(configs),
        "configs": {k: _plain(v) for k, v in configs.items()},
        "subgraph": subgraph,
        "metrics": metrics or {},
    }
    payload: dict[str, Any] = {**arrays, META_KEY: np.array(json.dumps(meta))}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(path)  # atomic, so a server never loads a half-written file
    return path


def read_meta(path: str | Path) -> dict[str, Any]:
    with np.load(path) as f:
        meta: dict[str, Any] = json.loads(str(f[META_KEY]))
    return meta


def load_checkpoint(
    path: str | Path,
    template: T,
    *,
    expect_subgraph: dict[str, Any] | None = None,
) -> tuple[T, dict[str, Any]]:
    """Load params shaped like `template`. Raises if keys, shapes, or the subgraph differ."""
    meta = read_meta(path)
    if expect_subgraph is not None:
        bad = subgraph_mismatch(meta["subgraph"] or {}, expect_subgraph)
        if bad:
            raise ValueError(
                f"{path}: trained on a different subgraph ({', '.join(bad)} differ). "
                "Use the subgraph files it was trained with "
                "(scripts/training/bundle_checkpoint.py)."
            )
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(path) as f:
        stored = {k for k in f.files if k != META_KEY}
        wanted = {jax.tree_util.keystr(k) for k, _ in leaves}
        if stored != wanted:
            raise ValueError(f"{path}: parameter keys differ: {sorted(stored ^ wanted)}")
        values = []
        for k, like in leaves:
            value = f[jax.tree_util.keystr(k)]
            if value.shape != np.shape(like):
                raise ValueError(f"{path}: {jax.tree_util.keystr(k)} has shape {value.shape}")
            values.append(jax.numpy.asarray(value))
    return jax.tree_util.tree_unflatten(treedef, values), meta
