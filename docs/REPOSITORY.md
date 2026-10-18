# Repository guide

Run repository commands from the repository root. `make setup` installs the Python
packages from `src/` in editable mode and installs frontend dependencies. Python
imports keep their existing names (`brain`, `fly_saber`, `training`, `server`);
`src` is a source directory, not an import prefix.

## File placement

| Directory | Contents |
| --- | --- |
| `src/brain/` | Connectome loading, subgraph construction, neural model and interventions |
| `src/fly_saber/` | JAX game environment, compound eye, experts and map parsing |
| `src/training/` | Training loops, policies, calibration, evaluation and checkpoint handling |
| `src/server/` | FastAPI app, catalogs, live sessions and protocol |
| `frontend/src/app/` | React entry point, workspace shell and global styles |
| `frontend/src/features/brain/` | Neuron data, rendering, live activity buffer and brain UI |
| `frontend/src/features/game/` | Game scene, fly model and motor animation |
| `frontend/src/features/session/` | Backend/protocol client, audio transport, state and session UI |
| `frontend/src/features/songs/` | Chart parsing, song downloads, catalog and song selection UI |
| `frontend/src/components/` | Shared UI primitives and presentation helpers |
| `frontend/src/lib/` | Shared utilities and theme |
| `frontend/worker/` | Cloudflare proxy and static asset hosting |
| `frontend/public/` | Committed browser assets and exported brain data |
| `scripts/` | Repository maintenance commands, grouped by purpose |
| `infra/docker/` | Container recipe, smoke-session client and normalized smoke map |
| `infra/deploy/` | Vultr and Cloudflare deployment scripts |
| `infra/slurm/` | HiPerGator environment, jobs and experiment launcher |
| `configs/` | Shared YAML configuration |
| `data/` | Downloaded/processed data and maps; large local files remain ignored |
| `bundles/` | Shipped checkpoints with their exact subgraphs and manifests |
| `results/` | Committed experiment logs and evaluation results |
| `tests/` | Python tests grouped by package |
| `frontend/tests/`, `frontend/e2e/` | Frontend unit tests and browser integration tests |
| `docs/project/` | Original project specification, team split and milestones |
| `docs/frontend/` | Frontend plans, animation and protocol integration notes |
| `docs/figures/` | Research figures and demo screenshots |

Put domain-specific frontend components in that feature's `components/` folder.
Keep reusable primitives in `frontend/src/components/ui/`. Use `@/` imports for
frontend application code; direct TypeScript imports used by Node tests keep
relative paths and explicit `.ts` extensions.

## Maintenance commands

The short commands remain in the root Makefile. The scripts can also run directly:

| Purpose | Command |
| --- | --- |
| Download connectome | `uv run python scripts/data/fetch_data.py` |
| Download demo maps | `uv run python scripts/data/fetch_maps.py` |
| Download training maps | `uv run python scripts/data/fetch_training_maps.py` |
| Build subgraph | `uv run python scripts/connectome/build_subgraph.py` |
| Build control variants | `uv run python scripts/connectome/build_variants.py` |
| Export browser anatomy | `uv run python scripts/web/export_web.py --bundle bundles/real_fast` |
| Export brain modes | `uv run python scripts/web/export_brains.py` |
| Benchmark brain | `uv run python scripts/training/bench_brain.py` |
| Evaluate checkpoints | `uv run python scripts/training/eval_checkpoints.py <checkpoint.npz>` |
| Bundle checkpoint | `uv run python scripts/training/bundle_checkpoint.py <checkpoint.npz> <output-dir>` |
| Plot learning curves | `uv run python scripts/training/plot_learning_curves.py` |

Scripts are also importable as modules, for example `scripts.web.export_web`.
They operate on root-level `configs/`, `data/`, `bundles/` and `frontend/public/`.
These runtime assets retain their paths to preserve checkpoint metadata, bundle
hashes, local datasets and deployment volumes. `fly_saber.config` resolves the
shared config directory relative to the source tree, including in the container.

## Development and deployment

- `make serve`: API and built frontend on port 8000.
- `make frontend`: frontend development server.
- `make clean`: remove generated build output, test reports and Python caches.
- `make check`: Python/frontend lint, types, builds, unit and browser tests.
- `make docker-build`: build using `infra/docker/Dockerfile` from the root context.
- `bash infra/deploy/deploy_vultr.sh`: rebuild and restart the backend on the server.
- `bash infra/deploy/go_live.sh`: deploy the backend and configure its Worker proxy.
- `sbatch --export=ALL infra/slurm/setup.sbatch`: prepare HiPerGator dependencies/data.

See [CI and deployment](CI_CD.md), [frontend setup](../frontend/README.md) and
[HiPerGator jobs](../infra/slurm/README.md) for details. Deployment scripts change
to the repository root before using paths. SLURM jobs are submitted from the root.

## Cleanup boundaries

`make clean` preserves installed dependencies, configuration, downloaded data,
checkpoints, bundles and committed result files. Temporary editor/patch files and
TypeScript build metadata are ignored by Git. Empty data download directories keep
`.gitkeep` files; populated directories and unused frontend scaffold directories do
not need them. The results report lives in `results/README.md`.
