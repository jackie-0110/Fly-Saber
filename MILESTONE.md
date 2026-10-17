# Fly Saber Milestones

What stands between us and the MVP: the fly playing Fly Saber on stage and at a public URL. Companion to `FLY_SABER_HANDOFF.md` (the spec) and `FLY_SABER_TEAM_SPLIT.md` (who owns what).

**For people and agents:** keep this file current. When you close an item, tick it in the same PR and name the PR. When you find a new blocker, add it under the right milestone with an owner. Status below is as of 2026-10-17, after #43.

## The MVP

The stage demo, on the public URL. The MVP is done when all four hold there:

1. The trained fly plays a song in the browser, audio in sync, score in the HUD. **Works locally**; public host pending (D).
2. A judge lesions the looming detectors or blindfolds the fly mid-song, and it visibly misses. **Server done**, UI missing (C1), looming lesion measured (C6).
3. The 3D neuron cloud lights up with live activity. **Not started** (C3).
4. Anyone picks any BeatSaver map and the fly plays it, including maps it has never seen. **Works locally**; brain picker UI pending (C5).

Plus the acceptance tests in handoff section 18 that still apply (F2).

Out of the MVP: real vs. scrambled Versus, the replay fallback, and the allowlisted-audio rule, all cut by the team on 2026-10-17. The rest of "If we have time" (Milestone E) waits too.

## Critical path

Work top to bottom. Items in the same step can run in parallel.

1. **Unblock:** BeatSaver play (C5), so #44 can merge; pick the stage song (below).
2. **Interactive demo:** C1, C3, C6.
3. **Public URL:** D3, D6, D7, in parallel with step 2.
4. **Ship:** F1 to F3.

Front-end work is split: Person 2 builds the client plumbing (session client and store: BeatSaver map loading, `set_interventions`, activity decoding), Person 3 builds the UI on top.

## Decisions needed

- **Stage song and the score we claim (all).** The only held-out allowlisted song plays to the fly's weak spots: `2dd26_expert` scores 16.4% live (`2dd26_expertplus` 21.6%). The best allowlisted map, `304be_expertplus`, scores 33.3% but is a training song. Options: lead with 304be labelled as a training song and show 2dd26 as "never heard"; or, once C5 lands, pick a held-out BeatSaver map that suits the fly.

## Done

- **Milestone A, a brain playing in the browser:** server plays checkpointed brains, per-slot state, `ready` with `n_neurons`/`cell_types` (A1, #28, #35); session client, server catalog, HUD score and events, brain picker (A2 to A5, #29); integration run with the trained `real` brain: live e2e passes and a full `2dd26_expert` plays at 29.7 Hz (A6, #29 + #35).
- **Milestone B, the trained fly in the server:** trained checkpoint, 32.9% held-out hits under Person 1's eval (B1, #30); bundled with its own subgraph (B2, #27, #30); server serves `real`, `shuffled_layers`, `shuffled_signs` from `bundles/` (B3, #35, #37). Always load a bundle's own `subgraph/`, never a local rebuild.
- **Lesion Mode on the server:** `set_interventions` applies lesion, opto, mood and eye controls live, per slot (C1 server, #36, #38).
- **Image has its data:** the Docker image ships `bundles/` and `frontend/dist`; maps and audio download into a volume via `scripts/fetch_maps.py` (D2, #42); CI plays a full session with `real` inside the container (on `p2/integration`, not yet on main).
- **Load:** per-session sim threads, `MAX_SESSIONS` cap (D4, #43).
- **Expert at 7.0 m/s:** 100% on all 10 demo maps, 2325 of 2325 notes (acceptance test 4).
- **Fly plays from vision:** with both eyes blinded it hits 0%.

## Milestone C: the must-ship demo features

- [ ] **C1. Lesion Mode UI** (Person 3). `SessionClient` sends `set_interventions` (PROTOCOL.md section 3; payload in section 4) and treats its `error` as non-fatal. Panel: lesion looming detectors, blind left/right eye, an arousal slider (mood rides the same message). `frontend/public/data/brain/controls.json` lists the groups and moods.
- [ ] **C3. Live neuron activity.** Pin the binary layout in PROTOCOL.md section 6 (Person 1); stream kinds 1 and 2 (Person 2); decode them and light the neuron cloud (Person 3). Neuron order matches `neurons.json` and `ready.cell_types`. Must hold 30 Hz on the public host (F2). Server side and client plumbing in #50: kinds 1 and 2 stream as int8 change from rest, layout pinned in PROTOCOL.md section 6, `SessionClient` `onActivity`. Left (Person 3): pass `onActivity` from `store.ts` and light `brain-scene.tsx`.
- [ ] **C5. The fly plays any BeatSaver map.** Server and store done: `POST /api/community/{key}` fetches a map's charts on demand and the browser plays the server's charts (#46); the normalizer reads v4 maps and play defaults to the `real` brain (p2/beatsaver-play). Left: a brain picker in the Community view (Person 3); an e2e test that a Community map starts a session with `real`, since #44 removed the live-session specs.
- [x] **C6. Looming lesion measurably lowers hit rate.** `real` (bc_950), held out, 64 × 10 s windows, after #36: untouched 32.9%, looming detectors lesioned 14.1%. Not selective: eye_left 7.1%, eye_right 7.6%, motion_detectors 2.1%, LC_all 6.1%, giant_fiber 16.8%; arousal −1 gives 4.8%, +1 gives 5.4%, so the arousal slider hurts in both directions.

## Milestone D: public URL

- [x] **D1. Hosting decision.** The front end is at https://flysaber.jackie-courtney.com, a Cloudflare Worker (#41) that proxies `/api/*` and `/ws/*` to its `BACKEND_URL` secret, so the browser stays same-origin and the server needs no CORS. Person 1 runs the backend on Vultr with `scripts/deploy_vultr.sh` (#42).
- [ ] **D3. Backend hostname and TLS** (Person 1). `docs/CI_CD.md` sets `BACKEND_URL=http://<server ip>:8000`; check that the Worker can fetch a bare IP and port, and if it can't, give the host a DNS name. Use a single-level name such as `flysaber-api.jackie-courtney.com`: Cloudflare's free certificate doesn't cover `api.flysaber.jackie-courtney.com`.
- **D5. Allowlisted audio only: dropped** (2026-10-17). This is a non-commercial demo; spec 8.4 no longer applies, and BeatSaver audio plays in the browser.
- [ ] **D6. Size the public host** (Person 1). The real brain runs about 2.1x real time on a laptop CPU, so a laptop keeps about two brain sessions at 30 Hz. Set `MAX_SESSIONS` to what the Vultr host keeps at 30 Hz with activity streaming.
- [ ] **D7. Fix the "Workers Builds: flysaber" check** (whoever set up Cloudflare). It fails on every PR.

## Milestone E: if we have time

- [ ] **E1. Replay fallback** (Person 3, Person 2 pairs). `scripts/record_replays.py` and a replay player that runs with the server off. Both are stubs.
- **Real vs. scrambled Versus.** The server already runs 2 slots and serves both controls; the front end would need two-slot handling and a split screen. #30 measured, held out: partners shuffled within layers 51.8%, real 31.4%, E/I shuffled 20.0%.
- **B4. Improve the trained fly.** Not approved. If it is: target up and lateral cuts and off-centre cells, for example by reweighting those classes in BC; longer clips won't help (whole songs and 5 s clips score the same).
- Fly's-eye view (eye stream kind 3 and `/api/eye_lattice`), opto and the remaining moods in the UI, Brain Explorer live traces (`/api/explorer`), Gemini narrator (first to cut), You vs. the Fly, curriculum and DAgger, PPO, matched MLP baseline, Fly Runner.

## Milestone F: ship

- [ ] **F1. README results with real numbers** (Person 3 owns `docs/RESULTS.md`): held-out hit rate, the looming lesion, the blindfold.
- [ ] **F2. Acceptance tests** (handoff section 18) pass on the public host, including 30 Hz with activity streaming. The replay and allowlisted-audio tests are dropped with E1 and D5.
- [ ] **F3. Rehearse the demo** (all).

## Risks to check

- The held-out song plays to the fly's weak spots. Held-out hit rate by note: bottom row middle lanes 41 to 49%, outer lanes 17 to 27%, middle row centre 0%; down cuts 46%, up 22%, left and right 2 to 4%. `2dd26_expert` alternates up and down cuts and puts 18% of its notes in the centre cells.
- Arbitrary BeatSaver maps are harder than the demo set: expect low scores on fast Expert+ maps, and a few seconds' compile the first time a new map size plays unless startup compiles every size (C5).
- The sim runs notes at 7.0 m/s while the preview uses each map's own speed. Hit timing matches (both use `time_sec`); only note spacing on screen differs.
