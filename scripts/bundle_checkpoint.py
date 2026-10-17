"""Package a checkpoint with the exact subgraph it was trained on, for the server.

Owner: Person 1. Usage (on HiPerGator, repo root):
    uv run python scripts/bundle_checkpoint.py \
        checkpoints/real/fast/seed0/bc_950.npz bundles/real
Then copy the folder off the cluster, for example:
    scp -r <gatorlink>@hpg.rc.ufl.edu:/blue/uf-sps/<gatorlink>/Fly-Saber/bundles/real .

The bundle holds the checkpoint, its log, config, and eval files, and subgraph/ (neurons,
edges, groups, meta). Load it with training.play_eval.load_trained(ckpt, subgraph_dir=...)
or point model.subgraph_dir at bundle/subgraph. Do not rebuild the subgraph elsewhere:
rebuilds can put different neurons in different slots.
"""

import sys
from pathlib import Path

from training.bundle import bundle

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"bundled into {bundle(Path(sys.argv[1]), Path(sys.argv[2]))}")
