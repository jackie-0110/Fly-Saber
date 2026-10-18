# training/

Spec section 12. Shared folder, one owner per file.

| Path | Owner | What |
|---|---|---|
| `bc.py` | Person 2 | Behavior cloning from the scripted expert, curriculum |
| `ppo.py` | Person 1 | PureJaxRL-style recurrent PPO from the BC checkpoint |
| `checkpoints.py` | Person 1 | Save and load with config hash |
| `evaluate.py` | Person 3 | Learning curves, lesion matrix, slider validation, held-out maps |
| `infra/slurm/` (repository root) | Person 1 | HiPerGator `sbatch` scripts |

Configs: `configs/train_bc.yaml`, `train_ppo.yaml`. Outputs go to `checkpoints/` and `runs/` (gitignored); figures to `docs/figures/`.
