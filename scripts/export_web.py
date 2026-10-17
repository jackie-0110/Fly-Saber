"""Export brain data (neurons, groups, eye columns, UI controls) into frontend/public/data/brain/.

Owner: Person 1. Usage: uv run python scripts/export_web.py
Rerun after scripts/build_subgraph.py whenever the subgraph changes, then commit.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from brain.config import SubgraphConfig, load_config
from brain.connectome.metadata import web_columns, web_controls, web_neurons
from brain.model.brain import eye_inputs
from brain.model.controls import load_controls
from fly_saber.eye import EyeConfig

OUT_DIR = Path("frontend/public/data/brain")


def main() -> None:
    cfg = load_config("configs/subgraph.yaml", SubgraphConfig)
    sub = Path(cfg.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    neurons = pd.read_parquet(sub / "neurons.parquet")
    (OUT_DIR / "neurons.json").write_text(json.dumps(web_neurons(neurons), separators=(",", ":")))
    shutil.copyfile(sub / "groups.json", OUT_DIR / "groups.json")
    eye = load_config("configs/eye.yaml", EyeConfig)
    columns, inputs = eye_inputs(neurons)
    table = web_columns(columns, inputs, eye.ommatidium_spacing_deg, eye.binocular_overlap_deg)
    (OUT_DIR / "columns.json").write_text(json.dumps(table, separators=(",", ":")))
    groups = json.loads((sub / "groups.json").read_text())
    ui = web_controls(load_controls(), groups)
    (OUT_DIR / "controls.json").write_text(json.dumps(ui, indent=1))
    for f in sorted(OUT_DIR.iterdir()):
        print(f"wrote {f} ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
