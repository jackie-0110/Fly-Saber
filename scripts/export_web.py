"""Export neurons, full directed wiring, groups, eye columns, and UI controls for the browser.

Owner: Person 1. Usage: uv run python scripts/export_web.py
For the committed trained model: uv run python scripts/export_web.py --bundle bundles/real_fast
Rerun after scripts/build_subgraph.py whenever the subgraph changes, then commit.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from brain.config import SubgraphConfig, load_config
from brain.connectome.metadata import web_columns, web_controls, web_edges, web_neurons
from brain.model.brain import eye_inputs
from brain.model.controls import load_controls
from fly_saber.eye import EyeConfig
from training.checkpoints import read_meta, subgraph_fingerprint, subgraph_mismatch

OUT_DIR = Path("frontend/public/data/brain")


def export_web(sub: Path, out: Path = OUT_DIR) -> None:
    """Keep all exported assets on the same subgraph and original neuron indices."""
    neurons = pd.read_parquet(sub / "neurons.parquet")
    edges = pd.read_parquet(sub / "edges.parquet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "neurons.json").write_text(json.dumps(web_neurons(neurons), separators=(",", ":")))
    (out / "edges.json").write_text(
        json.dumps(web_edges(neurons, edges, str(sub / "edges.parquet")), separators=(",", ":"))
    )
    shutil.copyfile(sub / "groups.json", out / "groups.json")
    eye = load_config("configs/eye.yaml", EyeConfig)
    columns, inputs = eye_inputs(neurons)
    table = web_columns(columns, inputs, eye.ommatidium_spacing_deg, eye.binocular_overlap_deg)
    (out / "columns.json").write_text(json.dumps(table, separators=(",", ":")))
    groups = json.loads((sub / "groups.json").read_text())
    ui = web_controls(load_controls(), groups)
    (out / "controls.json").write_text(json.dumps(ui, indent=1))
    for f in sorted(out.iterdir()):
        print(f"wrote {f} ({f.stat().st_size / 1e6:.2f} MB)")


def bundle_subgraph(bundle: Path) -> Path:
    """Refuse a bundle whose anatomy no longer matches its training checkpoint."""
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    sub = bundle / "subgraph"
    actual = subgraph_fingerprint(sub)
    checkpoint = read_meta(bundle / manifest["checkpoint"])
    for expected in (manifest["subgraph_fingerprint"], checkpoint["subgraph"]):
        bad = subgraph_mismatch(expected, actual)
        if bad:
            raise ValueError(f"Bundle subgraph mismatch: {', '.join(bad)}")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle", type=Path, help="Export the exact graph of a trained bundle")
    source.add_argument("--subgraph-dir", type=Path, help="Override configs/subgraph.yaml")
    args = parser.parse_args()
    if args.bundle:
        sub = bundle_subgraph(args.bundle)
    else:
        sub = args.subgraph_dir or Path(
            load_config("configs/subgraph.yaml", SubgraphConfig).out_dir
        )
    export_web(sub)


if __name__ == "__main__":
    main()
