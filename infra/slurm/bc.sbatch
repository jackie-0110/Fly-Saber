#!/bin/bash
# Behavior cloning on the connectome brain: copy the scripted expert, save a checkpoint.
# Owner: Person 1. Usage (from the repo root), overrides optional:
#   sbatch --export=ALL infra/slurm/bc.sbatch run_name=fast window_sec=5 brain_learning_rate=1e-4
# Output: checkpoints/<variant>/<run_name>/seed<seed>/bc_<update>.npz every 50 updates, plus
# bc_log.json and config.json. Defaults in configs/train_bc.yaml. Batch of runs: experiments.sh.
# Memory: about 2.5 GB of GPU memory per 64 envs (gradient checkpointing, configs/model.yaml remat).
#SBATCH --job-name=fly-bc
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

source infra/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -c "import jax; print('jax', jax.__version__, jax.devices())"
uv run python -m training.bc "$@"
