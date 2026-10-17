"""Package a checkpoint with the exact subgraph it was trained on.

Owner: Person 1. Used by scripts/bundle_checkpoint.py and scripts/export_brains.py.
"""

import json
import shutil
from pathlib import Path

from training.checkpoints import read_meta, subgraph_fingerprint, subgraph_mismatch

SUBGRAPH_FILES = ("neurons.parquet", "edges.parquet", "groups.json", "meta.json")


def bundle(checkpoint: Path, out: Path) -> Path:
    """Copy `checkpoint`, its log/config/eval, and its subgraph into `out`, with a manifest."""
    meta = read_meta(checkpoint)
    src = Path(meta["configs"]["model"]["subgraph_dir"])
    bad = subgraph_mismatch(meta["subgraph"] or {}, subgraph_fingerprint(src))
    if bad:
        raise SystemExit(f"{src} does not match the checkpoint ({', '.join(bad)}); not bundling")
    (out / "subgraph").mkdir(parents=True, exist_ok=True)
    for name in SUBGRAPH_FILES:
        shutil.copyfile(src / name, out / "subgraph" / name)
    shutil.copyfile(checkpoint, out / checkpoint.name)
    for extra in ("bc_log.json", "config.json", f"eval_{checkpoint.stem}.json"):
        if (checkpoint.parent / extra).exists():
            shutil.copyfile(checkpoint.parent / extra, out / extra)
    manifest = {
        "checkpoint": checkpoint.name,
        "source": str(checkpoint),
        "variant": meta["variant"],
        "step": meta["step"],
        "metrics": meta["metrics"],
        "subgraph_fingerprint": subgraph_fingerprint(out / "subgraph"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return out


def bundle_view_only(subgraph: Path, out: Path, variant: str) -> Path:
    """A brain with no trained checkpoint: its subgraph and a manifest, for showing wiring."""
    (out / "subgraph").mkdir(parents=True, exist_ok=True)
    for name in SUBGRAPH_FILES:
        shutil.copyfile(subgraph / name, out / "subgraph" / name)
    manifest = {
        "checkpoint": None,
        "variant": variant,
        "subgraph_fingerprint": subgraph_fingerprint(out / "subgraph"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return out
