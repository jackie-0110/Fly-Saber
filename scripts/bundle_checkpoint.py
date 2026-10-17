"""Package a checkpoint with the exact subgraph it was trained on, for the server.

Owner: Person 1. Usage (on HiPerGator, repo root):
    uv run python scripts/bundle_checkpoint.py \
        checkpoints/real/fast/seed0/bc_950.npz bundles/real_fast
Then copy the folder off the cluster, for example:
    scp -r <gatorlink>@hpg.rc.ufl.edu:/blue/uf-sps/<gatorlink>/Fly-Saber/bundles/real_fast .

The bundle holds the checkpoint, its log, config, and eval files, and subgraph/ (neurons,
edges, groups, meta). Load it with training.play_eval.load_trained(ckpt, subgraph_dir=...)
or point model.subgraph_dir at bundle/subgraph. Do not rebuild the subgraph elsewhere:
rebuilds can put different neurons in different slots.
"""

import json
import shutil
import sys
from pathlib import Path

from training.checkpoints import read_meta, subgraph_fingerprint, subgraph_mismatch

SUBGRAPH_FILES = ("neurons.parquet", "edges.parquet", "groups.json", "meta.json")


def bundle(checkpoint: Path, out: Path) -> Path:
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"bundled into {bundle(Path(sys.argv[1]), Path(sys.argv[2]))}")
