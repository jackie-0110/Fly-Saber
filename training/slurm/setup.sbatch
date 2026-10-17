#!/bin/bash
# One-time setup on HiPerGator: install deps with GPU JAX, download the
# connectome (~1.1 GB), build the subgraph. CPU only. Owner: Person 1.
# Usage (from the repo root): sbatch --export=ALL training/slurm/setup.sbatch
#SBATCH --job-name=fly-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source training/slurm/env.sh
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cuda
uv run python scripts/fetch_data.py
uv run python scripts/build_subgraph.py
echo "setup done"
