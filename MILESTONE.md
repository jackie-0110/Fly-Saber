# Fly Saber Milestones

What stands between us and the fly playing Fly Saber on stage, in the order it has to happen. Companion to `FLY_SABER_HANDOFF.md` (the spec) and `FLY_SABER_TEAM_SPLIT.md` (who owns what).

**For people and agents:** keep this file current. When you close an item, tick it in the same PR and name the PR. When you find a new blocker, add it under the right milestone with an owner. Status below is as of 2026-10-17.

## Where we are

- The server runs real-time sessions (30 Hz, 1 or 2 slots for Versus) on the real env, played by the scripted expert. PR #28 adds checkpointed brains.
- Person 1 has trained a demo fly (PR #30): `bundles/real_fast/`, real wiring, brain trained, 31.4% of notes hit on held-out songs (28.4% on training songs). It is bundled with the exact subgraph it was trained on.
- PR #29 connects the browser to single-player sessions: server songs and brain selection, audio-clock synchronization, live saber motion, score and note events. Desktop/mobile integration passes with FastAPI serving the built frontend and the scripted expert playing an original silent fixture. Trained-bundle loading remains B3; activity, interventions, Versus and replay remain Milestone C.

## Milestone A: a brain visibly playing a song in the browser

The first end-to-end moment. Everything else builds on it.

- [ ] **A1. Server plays checkpointed brains.** PR #28 (Person 2): `/api/brains` lists `real` and `rewired` when their checkpoints exist; each slot carries its own brain state; `ready` reports `n_neurons` and `cell_types`.
- [x] **A2. Session client in the front end** (Person 3), PR #29. Open `WS /ws/session`; send `start {game, brains: [...], map_id, seed}`, `play {song_t}` when audio starts, `clock {song_t}` about once a second; handle `ready`, `frame`, `event`, `ended`, `error` (`docs/PROTOCOL.md` section 3). Attach PR #29's motion adapter to `frame.render_state.sabers`.
- [x] **A3. Songs from the server catalog** (Person 3, Person 2 owns the contract), PR #29. The fly can only play maps the server knows: live play picks from `GET /api/maps` and plays `GET /api/maps/{id}/audio`. Charts loaded from BeatSaver in the browser cannot be played by the fly.
- [x] **A4. Score and note events in the HUD** (Person 3), PR #29: `render_state.score`, and `event {kind, note_id}` to slice or remove notes.
- [x] **A5. Brain picker** (Person 3), PR #29, from `GET /api/brains`. Only advertised players are offered; prefer `real` when available.
- [x] **A6. One integration run** (Person 2 and Person 3), PR #29: FastAPI serving `frontend/dist`, browser audio and real env/expert simulating the same `map_id`, verified on desktop/mobile with an original silent fixture. Repeat with a trained brain and demo audio after B3 and map staging. Opt-in regression: `FLY_SABER_INTEGRATION_URL=http://127.0.0.1:8000 npx playwright test live-integration`.

## Milestone B: the trained fly in the server

- [x] **B1. A trained checkpoint.** PR #30 (Person 1): `bundles/real_fast/bc_950.npz`, 31.4% held-out hits. Full results in `results/README.md`.
- [x] **B2. Checkpoint bundled with its subgraph.** PR #27 adds `scripts/bundle_checkpoint.py`, a deterministic subgraph build, and a content-hash fingerprint; PR #30 ships the bundle. Subgraph rebuilds on different machines differ (19,462 vs 19,463 neurons), so **always load a bundle's own `subgraph/`, never a local rebuild.**
- [x] **B3. Server loads bundles** (Person 2). `/api/brains` lists each mode in `configs/brains.yaml` with a trained `bundles/<id>/`, loaded with `training.play_eval.load_trained` from the bundle's own subgraph; a bundle that fails to load is skipped with a log line. With the bundles from #37 it serves `real`, `shuffled_layers` and `shuffled_signs`.
- [ ] **B4. Improve the trained fly** (Person 1 runs, Person 2 owns BC). Readout-only (33.9% held-out) still beats brain training (31.4%). Not blocking: the fly already plays from vision.

## Milestone C: the must-ship demo features

- [ ] **C1. Lesion Mode.** Server: `set_interventions` builds per-slot interventions with `brain.model.controls.from_controls` (Person 2). Front end: lesion panel (Person 3). Live eye controls from PR #26 can ride the same message.
- [ ] **C3. Live neuron activity.** Pin the binary layout in `docs/PROTOCOL.md` section 6 (Person 1); stream kinds 1 and 2 (Person 2); light up the neuron cloud (Person 3). PR #30 re-exports `frontend/public/data/brain/*.json` from the trained fly's subgraph, so neuron indices line up.
- [ ] **C4. Replay fallback** (Person 3, Person 2 pairs). `scripts/record_replays.py` and a replay player that runs with the server off. Both are stubs.

## Milestone D: public URL

- [ ] **D1. Hosting decision** (Person 2, with whoever set up Cloudflare). A Cloudflare project "flysaber" builds on every commit, configured outside the repo. The front end calls relative `/api` and `/ws` paths, so a front end hosted apart from the Python server needs a configurable backend URL. Default plan: one Vultr host serves both.
- [ ] **D2. Image has its data** (Person 2). Mount or bake in `data/maps` (normalized maps, allowlisted audio), `bundles/`, and `frontend/dist`.
- [ ] **D3. Domain and HTTPS** (Person 2; needs the team's Vultr and GoDaddy accounts).
- [x] **D4. Load** (Person 2). Each session's sim work runs on its own thread, so the event loop never waits on JAX and sessions use separate cores. `MAX_SESSIONS` (default 2: two Versus sessions are 4 brain slots, about what a laptop CPU keeps at real time) caps concurrent sessions; set it to the host's capacity.
- [ ] **D5. Allowlisted audio only on the public URL** (Person 3, spec 8.4): the in-browser BeatSaver search must not play non-allowlisted audio there.

## Risks to check

- The only held-out allowlisted song (2dd26) has only Expert and Expert+ difficulties: a hard song for "plays a song it has never heard".
- Expert near 100 on every demo map (acceptance test 4) was measured at 8 m/s; note speed is now 7.0 m/s. Re-measure.
- The sim runs notes at 7.0 m/s while the preview uses each map's own speed. Hit timing matches (both use `time_sec`); only note spacing on screen differs.

## Can wait

Fly's-eye view (eye stream and `/api/eye_lattice`), mood sliders and opto in the UI, curriculum and DAgger, PPO, matched MLP baseline, Gemini narrator (first to cut).
