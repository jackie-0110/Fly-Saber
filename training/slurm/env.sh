# Sourced by every Fly Saber SLURM job and by setup.sh (owner: Person 1).
# Keeps uv's caches and Python installs on /blue: home is small and fills up fast.
set -euo pipefail

# FLY_BLUE is the folder that contains the Fly-Saber repo. If unset, use the parent of the
# directory the job was submitted from (sbatch from the repo root, as the README says).
if [[ -z "${FLY_BLUE:-}" ]]; then
    FLY_BLUE="$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")"
fi
export FLY_BLUE
if [[ ! -d "$FLY_BLUE/Fly-Saber" ]]; then
    echo "env.sh: no Fly-Saber repo in $FLY_BLUE; submit from the repo root or export FLY_BLUE" >&2
    exit 1
fi
if [[ "$FLY_BLUE" == /home/* ]]; then
    echo "env.sh: warning: $FLY_BLUE is in home; uv caches may exceed the home quota, use /blue" >&2
fi
export UV_CACHE_DIR="$FLY_BLUE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$FLY_BLUE/.local/uv-python"
export PATH="$HOME/.local/bin:$PATH"

# Let JAX allocate GPU memory as needed instead of grabbing 75% up front.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$FLY_BLUE/Fly-Saber"
mkdir -p logs
