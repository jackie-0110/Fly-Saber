# Architecture

Connectome subgraph (Person 1) -> brain model (Person 1) <- fly eye + Fly Saber env (Person 2) -> server (Person 2) -> web front end (Person 3).

Fill in with a diagram once the M2 end-to-end path works.

## Conventions

| Concern | Choice | Why |
|---|---|---|
| JAX pytrees (env state, params, interventions) | frozen dataclasses registered with `jax.tree_util.register_dataclass` | built into JAX, no flax dependency, works under `jit` and `vmap` |
| YAML configs | frozen dataclasses loaded with `pyyaml` | hashable, so they can be `jit` static args |
| WebSocket and REST messages | pydantic models, discriminated union on `type` | ships with FastAPI; exports JSON Schema for the front end |
| Live session compute | CPU; move to a GPU instance only if the M2 benchmark cannot hold 30 Hz | spec 15 |
