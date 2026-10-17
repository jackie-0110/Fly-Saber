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
sbatch --export=ALL training/slurm/setup.sbatch   # uv + GPU JAX + data + subgraph, ~15 min
```

## Check the GPU

```bash
sbatch --export=ALL training/slurm/bench.sbatch                          # L4 (hpg-turin)
sbatch --export=ALL --partition=hpg-b200 training/slurm/bench.sbatch     # B200
squeue --me                                                              # wait for it
cat logs/fly-bench-<jobid>.out
```

Success: `jax ... [CudaDevice(id=0)]` and frames per second far above the laptop's ~1,800.

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
- `jax ... [CpuDevice(id=0)]` in the bench log: GPU JAX did not install. Run `source training/slurm/env.sh && uv sync --extra cuda` inside a GPU job, check `nvidia-smi` works there.
- Disk quota errors: uv caches go to `$FLY_BLUE/.cache` through `env.sh`; check `home_quota` and `blue_quota`.
- Job pending a long time: `squeue --me` shows the reason; try the other GPU partition.
