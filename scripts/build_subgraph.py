"""Build data/processed/{dataset}_subgraph: neurons, edges, groups, meta. Refuses to
replace an existing build without --force (checkpoints depend on its exact neuron order).

Owner: Person 1. Usage: uv run python scripts/build_subgraph.py [configs/subgraph.yaml]
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from brain.config import SubgraphConfig, load_config
from brain.connectome.groups import build_groups
from brain.connectome.load_malecns import load_edges, load_neurons
from brain.connectome.subgraph import build_subgraph


def main(config_path: str = "configs/subgraph.yaml", force: bool = False) -> None:
    cfg = load_config(config_path, SubgraphConfig)
    out = Path(cfg.out_dir)
    if (out / "neurons.parquet").exists() and not force:
        # Trained checkpoints, the server, and the front end all index neurons by this
        # build. A rebuild can differ (it did across machines), so never replace it silently.
        print(f"{out} already exists; keeping it (checkpoints were trained on it). Use --force.")
        return
    out.mkdir(parents=True, exist_ok=True)

    print("loading neurons ...")
    neurons = load_neurons(cfg.raw_dir)
    print(f"  {len(neurons)} traced typed neurons")
    print("loading edges (first run filters 152M rows, about a minute) ...")
    edges = load_edges(cfg.raw_dir, neurons.index, cfg.min_synapses)
    print(f"  {len(edges)} edges with >= {cfg.min_synapses} synapses")

    sub = build_subgraph(neurons, edges, cfg)
    groups = build_groups(sub.neurons)

    sub.neurons.to_parquet(out / "neurons.parquet")
    sub.edges.to_parquet(out / "edges.parquet")
    (out / "groups.json").write_text(json.dumps(groups))
    meta = {
        "source": "MaleCNS v1.0 flat connectome, minconf 0.5",
        "n_neurons": len(sub.neurons),
        "n_edges": len(sub.edges),
        "n_synapses": int(sub.edges["weight"].sum()),
        "roles": sub.neurons["role"].value_counts().to_dict(),
        "signs": {str(k): int(v) for k, v in sub.edges["sign"].value_counts().items()},
        "positions_inherited": int(sub.neurons["position_inherited"].sum()),
        "positions_missing": int(sub.neurons["x"].isna().sum()),
        "config": asdict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2))
    print("groups:", {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if a != "--force"]
    main(paths[0] if paths else "configs/subgraph.yaml", force="--force" in sys.argv[1:])
