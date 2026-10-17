#!/bin/bash
# Behavior cloning on the connectome brain: copy the scripted expert, save a checkpoint.
# Owner: Person 1. Usage (from the repo root):
#   sbatch --export=ALL training/slurm/bc.sbatch
# Output: checkpoints/real/seed0/bc_<n_updates>.npz and bc_log.json; settings in configs/train_bc.yaml.
# Memory: about 2.5 GB of GPU memory per 64 envs (gradient checkpointing, configs/model.yaml remat).
#SBATCH --job-name=fly-bc
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

source training/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -c "import jax; print('jax', jax.__version__, jax.devices())"
uv run python -m training.bc
