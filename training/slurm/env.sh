# Sourced by every FlyArcade SLURM job and by setup.sh (owner: Person 1).
# Keeps uv's caches and Python installs on /blue: home is small and fills up fast.
set -euo pipefail

: "${FLY_BLUE:?set FLY_BLUE to your /blue folder, e.g. export FLY_BLUE=/blue/mygroup/mygatorlink}"
export UV_CACHE_DIR="$FLY_BLUE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$FLY_BLUE/.local/uv-python"
export PATH="$HOME/.local/bin:$PATH"

# Let JAX allocate GPU memory as needed instead of grabbing 75% up front.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$FLY_BLUE/Fly-Saber"
mkdir -p logs
