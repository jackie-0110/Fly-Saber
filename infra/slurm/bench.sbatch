#!/bin/bash
# Check JAX sees the GPU, then benchmark the brain at training batch sizes.
# Owner: Person 1. Usage: sbatch --export=ALL infra/slurm/bench.sbatch
# B200 instead of L4: sbatch --export=ALL --partition=hpg-b200 infra/slurm/bench.sbatch
#SBATCH --job-name=fly-bench
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source infra/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "import jax; print('jax', jax.__version__, jax.devices())"
uv run python scripts/training/bench_brain.py 64 1024 4096
