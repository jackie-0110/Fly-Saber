# HiPerGator (UF) setup

Owner: Person 1. Everything below runs on HiPerGator. Do not run computation on the login node; submit jobs.

## Once

```bash
ssh <gatorlink>@hpg.rc.ufl.edu                 # Duo prompt
id -gn; groups                                 # your group(s), e.g. mygroup
slurmInfo                                      # CPU and GPU limits per group
export FLY_BLUE=/blue/<group>/<gatorlink>      # add this line to ~/.bashrc
cd $FLY_BLUE
```

The repo is private, so give HiPerGator its own GitHub key:

```bash
ssh-keygen -t ed25519 -C "hipergator" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub     # add at github.com/settings/keys
ssh -T git@github.com         # should greet you by username
git clone git@github.com:jackie-0110/Fly-Saber.git && cd Fly-Saber
mkdir -p logs
sbatch --export=ALL infra/slurm/setup.sbatch   # uv + GPU JAX + data + subgraph, ~15 min
```

## Check the GPU

```bash
sbatch --export=ALL infra/slurm/bench.sbatch                          # L4 (hpg-turin)
sbatch --export=ALL --partition=hpg-b200 infra/slurm/bench.sbatch     # B200
squeue --me                                                              # wait for it
cat logs/fly-bench-<jobid>.out
```

Success: `jax ... [CudaDevice(id=0)]` and frames per second far above the laptop's ~1,800.

## Train (behavior cloning)

Needs setup to have run on the current code (it also fetches the song maps). Already set up from an older checkout? Run `uv run python scripts/data/fetch_maps.py` once inside a job, or resubmit `setup.sbatch`.

```bash
sbatch --export=ALL infra/slurm/bc.sbatch
tail -f logs/fly-bc-<jobid>.out     # one row every 50 updates
```

Each row: `loss` should fall; `head_accuracy` is [red position, red direction, blue position, blue direction]. Direction starts near 95% only because "no cut" is almost always right, so watch position accuracy and `hit_rate` (greedy play, fraction of notes hit). Output: `checkpoints/real/seed0/bc_2000.npz` and `bc_log.json`. Settings: `configs/train_bc.yaml`. GPU memory is about 2.5 GB per 64 envs.

## Experiments in parallel

```bash
git pull                                              # this branch or main
sbatch --export=ALL --job-name=fly-prep --cpus-per-task=2 --mem=8gb --time=00:20:00 \
  --output=logs/%x-%j.out --wrap "source infra/slurm/env.sh && uv run python scripts/connectome/build_variants.py"
bash infra/slurm/experiments.sh                    # after fly-prep finishes; 5 jobs, one L4 each
```

`experiments.sh` submits: `fast` (real wiring, brain lr 1e-4), `readout` (brain frozen), `rewired` and `signs` (control brains), `fast-s1` (second seed). Shared: 5 s windows, 256 games per update, 1000 updates, a checkpoint every 50. Results land in `checkpoints/<variant>/<run_name>/seed<seed>/` (`bc_50.npz`, `bc_100.npz`, ..., `bc_log.json`, `config.json`); logs in `logs/fly-<name>-<jobid>.out`.

One-off runs take overrides: `sbatch --export=ALL infra/slurm/bc.sbatch run_name=mytest window_sec=5 brain_learning_rate=1e-4 model.freeze_brain=true`. Bare keys set `configs/train_bc.yaml` fields; `model.`, `saber.`, `eye.` prefixes set those configs.

## Every time

```bash
cd $FLY_BLUE/Fly-Saber && git pull
squeue --me            # your jobs
scancel <jobid>        # stop one
sacct -j <jobid>       # what happened to a finished job
```

If you have more than one group, add `--account=<group> --qos=<group>` to `sbatch`.

## Troubleshooting

- `env.sh: no Fly-Saber repo in ...`: submit from the repo root (`cd $FLY_BLUE/Fly-Saber`), or export `FLY_BLUE` (the folder containing the repo) and resubmit with `--export=ALL`.
- `DependencyNeverSatisfied` on the bench job: the setup job failed. `scancel` the bench job, read `logs/fly-setup-<jobid>.out`, fix, and resubmit both.
- `jax ... [CpuDevice(id=0)]` in the bench log: GPU JAX did not install. Run `source infra/slurm/env.sh && uv sync --extra cuda` inside a GPU job, check `nvidia-smi` works there.
- Disk quota errors: uv caches go to `$FLY_BLUE/.cache` through `env.sh`; check `home_quota` and `blue_quota`.
- Job pending a long time: `squeue --me` shows the reason; try the other GPU partition.
