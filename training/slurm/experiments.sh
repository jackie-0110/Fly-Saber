#!/bin/bash
# Submit the BC experiment batch, one L4 job each, all in parallel (owner: Person 1).
# Usage (from the repo root): bash training/slurm/experiments.sh [extra key=value ...]
# Needs the control brains: uv run python scripts/build_variants.py (setup.sbatch runs it).
# Results: checkpoints/<variant>/<run_name>/seed<seed>/, one row per 50 updates in bc_log.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

# Shared settings: 5 s windows (reacting to a block takes well under a second), 256 games per
# update (about 10 GB of GPU memory), brain at 1e-4 (stable locally), checkpoint every 50.
COMMON="window_sec=5 batch_size=256 n_updates=1000 checkpoint_every=50 $*"
FAST="brain_learning_rate=0.0001"
REWIRED="variant=rewired model.subgraph_dir=data/processed/malecns_subgraph_rewired"
SIGNS="variant=sign_shuffled model.subgraph_dir=data/processed/malecns_subgraph_sign_shuffled"

submit() {  # job name, then overrides
    local name=$1; shift
    sbatch --export=ALL --job-name="fly-$name" training/slurm/bc.sbatch "$@" | sed "s/$/  ($name)/"
}

# 1. Real wiring, brain trains: the main fly.
submit fast run_name=fast $FAST $COMMON
# 2. Real wiring, readout only: does training the brain itself help?
submit readout run_name=readout_only brain_learning_rate=0.0 model.freeze_brain=true $COMMON
# 3. Rewired control (same degrees, scrambled partners): does the real wiring matter?
submit rewired run_name=fast $FAST $REWIRED $COMMON
# 4. Sign-shuffled control (real wiring, excitatory/inhibitory labels scrambled).
submit signs run_name=fast $FAST $SIGNS $COMMON
# 5. Second seed of the main fly, for error bars.
submit fast-s1 run_name=fast seed=1 $FAST $COMMON
