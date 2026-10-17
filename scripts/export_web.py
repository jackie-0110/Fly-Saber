"""Export brain data for the front end into frontend/public/data/brain/.

Owner: Person 1. Usage: uv run python scripts/export_web.py
Rerun after scripts/build_subgraph.py whenever the subgraph changes, then commit.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from brain.config import SubgraphConfig, load_config
from brain.connectome.metadata import web_neurons

OUT_DIR = Path("frontend/public/data/brain")


def main() -> None:
    cfg = load_config("configs/subgraph.yaml", SubgraphConfig)
    sub = Path(cfg.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    neurons = pd.read_parquet(sub / "neurons.parquet")
    (OUT_DIR / "neurons.json").write_text(json.dumps(web_neurons(neurons), separators=(",", ":")))
    shutil.copyfile(sub / "groups.json", OUT_DIR / "groups.json")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"wrote {f} ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
