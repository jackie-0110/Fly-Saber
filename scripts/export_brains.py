"""Export every brain mode for the demo: server bundles and front-end mode files.

Owner: Person 1. Run on HiPerGator from the repo root (it needs the trained checkpoints
and the subgraphs they were trained on):
    uv run python scripts/export_brains.py

Writes, per mode in configs/brains.yaml:
  bundles/<id>/                         checkpoint + exact subgraph (view-only: subgraph only)
  frontend/public/data/brains/<id>.json  per-neuron signs, strongest connections per layer pair
  frontend/public/data/brains/index.json labels, descriptions, play results, wiring stats
All modes share the real brain's neurons, so the front end keeps one neurons.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from brain.connectome.metadata import edge_sample, wiring_stats
from training.bundle import bundle, bundle_view_only

OUT = Path("frontend/public/data/brains")


@dataclass(frozen=True)
class BrainsConfig:
    real_subgraph: str
    edges_per_block: int
    modes: list[dict[str, Any]]


def play_results(checkpoint: Path) -> dict[str, Any] | None:
    """Held-out and training hit rates from eval_<ckpt>.json, else the last training row."""
    eval_file = checkpoint.with_name(f"eval_{checkpoint.stem}.json")
    if eval_file.exists():
        e = json.loads(eval_file.read_text())
        return {
            split: round(e[split]["untouched"]["hit_rate"], 4)
            for split in ("heldout", "train")
            if split in e
        }
    log = checkpoint.with_name("bc_log.json")
    if log.exists():
        return {"train_during_training": round(json.loads(log.read_text())[-1]["hit_rate"], 4)}
    return None


def main() -> None:
    cfg = BrainsConfig(**yaml.safe_load(Path("configs/brains.yaml").read_text()))
    real = Path(cfg.real_subgraph)
    real_neurons = pd.read_parquet(real / "neurons.parquet")
    real_edges = pd.read_parquet(real / "edges.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for mode in cfg.modes:
        sub = Path(mode["subgraph"])
        neurons = pd.read_parquet(sub / "neurons.parquet")
        edges = pd.read_parquet(sub / "edges.parquet")
        if neurons["body_id"].tolist() != real_neurons["body_id"].tolist():
            raise SystemExit(f"{sub} has different neurons than {real}; rebuild its variants")
        ckpt = Path(mode["checkpoint"]) if mode.get("checkpoint") else None
        playable = ckpt is not None and ckpt.exists()
        if mode.get("checkpoint") and not playable:
            print(f"  {mode['id']}: {ckpt} not found, exporting as view-only")
        out = Path("bundles") / mode["id"]
        if playable:
            bundle(ckpt, out)  # type: ignore[arg-type]
        else:
            bundle_view_only(sub, out, mode["id"])
        stats = wiring_stats(neurons, edges, real_edges)
        (OUT / f"{mode['id']}.json").write_text(
            json.dumps(
                {
                    "id": mode["id"],
                    "sign": neurons["sign"].astype(int).tolist(),
                    "edges_columns": ["pre", "post", "weight", "sign"],
                    "edges": edge_sample(neurons, edges, cfg.edges_per_block),
                },
                separators=(",", ":"),
            )
        )
        index.append(
            {
                "id": mode["id"],
                "label": mode["label"],
                "description": mode["description"],
                "playable": playable,
                "bundle": str(out),
                "play": play_results(ckpt) if playable else None,  # type: ignore[arg-type]
                "wiring": stats,
            }
        )
        print(
            f"{mode['id']:16s} playable={playable!s:5s} "
            f"{stats['direct_eye_to_motor_connections']:5d} direct eye->motor, "
            f"median {stats['eye_to_motor_synapses_median']} synapses, "
            f"{stats['connections_shared_with_real']:.1%} shared with real"
        )
    (OUT / "index.json").write_text(json.dumps({"schema_version": 1, "modes": index}, indent=1))
    print(f"wrote {OUT}/index.json and {len(index)} mode files")


if __name__ == "__main__":
    main()
