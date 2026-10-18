#!/bin/bash
# One-time setup on HiPerGator: install deps with GPU JAX, download the
# connectome (~1.1 GB), build the subgraph, fetch the song maps. CPU only. Owner: Person 1.
# Usage (from the repo root): sbatch --export=ALL infra/slurm/setup.sbatch
#SBATCH --job-name=fly-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

source infra/slurm/env.sh
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cuda
uv run python scripts/data/fetch_data.py
uv run python scripts/connectome/build_subgraph.py
uv run python scripts/data/fetch_maps.py            # allowlisted demo songs (with audio)
uv run python scripts/data/fetch_training_maps.py   # ~150 BeatSaver charts, the BC training data
uv run python scripts/connectome/build_variants.py        # rewired and sign_shuffled control brains
echo "setup done"
