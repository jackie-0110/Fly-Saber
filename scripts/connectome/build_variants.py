"""Build control brains (rewired, rewired_layered, sign_shuffled) next to the real subgraph.

Owner: Person 1. Usage: uv run python scripts/connectome/build_variants.py
Writes data/processed/malecns_subgraph_<name>/ with the same neurons and groups.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brain.config import load_config
from brain.connectome.variants import rewire, rewire_layered, shuffle_signs


@dataclass(frozen=True)
class VariantsConfig:
    subgraph_dir: str
    variants: list[str]
    seed: int


def main() -> None:
    cfg = load_config("configs/variants.yaml", VariantsConfig)
    src = Path(cfg.subgraph_dir)
    neurons = pd.read_parquet(src / "neurons.parquet")
    edges = pd.read_parquet(src / "edges.parquet")
    meta = json.loads((src / "meta.json").read_text())
    for name in cfg.variants:
        rng = np.random.default_rng(cfg.seed)
        if name == "rewired":
            v_neurons, v_edges = neurons, rewire(edges, rng)
        elif name == "rewired_layered":
            v_edges, stats = rewire_layered(edges, neurons, rng)
            v_neurons = neurons
            print(f"  {name}: {stats['blocks']} blocks, {stats['kept_edges']} edges kept as is")
        elif name == "sign_shuffled":
            v_neurons, v_edges = shuffle_signs(neurons, edges, rng)
        else:
            raise ValueError(f"unknown variant {name!r}")
        out = Path(f"{cfg.subgraph_dir}_{name}")
        out.mkdir(parents=True, exist_ok=True)
        v_neurons.to_parquet(out / "neurons.parquet")
        v_edges.to_parquet(out / "edges.parquet")
        shutil.copyfile(src / "groups.json", out / "groups.json")
        (out / "meta.json").write_text(
            json.dumps({**meta, "variant": name, "variant_seed": cfg.seed}, indent=2)
        )
        print(f"wrote {out}: {len(v_neurons)} neurons, {len(v_edges)} edges")


if __name__ == "__main__":
    main()
