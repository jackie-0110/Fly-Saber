# Architecture

`src/brain/` connectome subgraph and model (Person 1) <- `src/fly_saber/` fly eye and env (Person 2), trained by `src/training/` -> `src/server/` (Person 2) -> `frontend/` (Person 3).

See [Repository guide](../development/REPOSITORY.md) for the directory layout, import rules and maintenance commands.

## Conventions

| Concern | Choice | Why |
|---|---|---|
| JAX pytrees (env state, params, interventions) | frozen dataclasses registered with `jax.tree_util.register_dataclass` | built into JAX, no flax dependency, works under `jit` and `vmap` |
| YAML configs | frozen dataclasses loaded with `pyyaml` | hashable, so they can be `jit` static args |
| WebSocket and REST messages | pydantic models, discriminated union on `type` | ships with FastAPI; exports JSON Schema for the front end |
| Live session compute | CPU; move to a GPU instance only if the M2 benchmark cannot hold 30 Hz | spec 15 |
