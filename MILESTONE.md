# Fly Saber Milestones

What stands between us and the MVP: the fly playing Fly Saber on stage and at a public URL. Companion to `FLY_SABER_HANDOFF.md` (the spec) and `FLY_SABER_TEAM_SPLIT.md` (who owns what).

**For people and agents:** keep this file current. When you close an item, tick it in the same PR and name the PR. When you find a new blocker, add it under the right milestone with an owner. Status below is as of 2026-10-17, after #43.

## The MVP

The spec's must-ship tier (handoff section 1), as the stage demo shows it. The MVP is done when all six hold on the public URL:

1. The trained fly plays a demo song in the browser, audio in sync, score in the HUD. **Works locally**; public host pending (D).
2. A judge lesions the looming detectors or blindfolds the fly mid-song, and it visibly misses. **Server done**, UI missing (C1), looming effect unmeasured (C6).
3. Real and scrambled brains play the same song side by side. **Server done**, UI missing (C2).
4. The 3D neuron cloud lights up with live activity. **Not started** (C3).
5. A replay of each demo plays with the server off. **Not started** (C4).
6. The public URL plays only allowlisted audio. **Not started** (D5).

Plus the acceptance tests in handoff section 18 (E2).

Out of the MVP, in "Can wait": narrator, fly's-eye view, opto, Brain Explorer live traces, You vs. the Fly, retraining the fly (B4).

## Critical path

Work top to bottom. Items in the same step can run in parallel.

1. **Unblock:** keep the live song picker (C5); make the two decisions below.
2. **Interactive demo:** C1, C2, C3, C6.
3. **Public URL:** D3, D5, D6, in parallel with step 2.
4. **Safety net and ship:** C4, then E1 to E3.

## Decisions needed

- **Versus control (all).** #30 measured, held out: partners shuffled within layers 51.8%, real 31.4%, E/I shuffled 20.0%. Both controls are bundled (`shuffled_layers`, `shuffled_signs`). Pick one for the stage and decide what the pitch claims.
- **Stage song and the score we claim (all).** The only held-out allowlisted song plays to the fly's weak spots: `2dd26_expert` scores 16.4% live (`2dd26_expertplus` 21.6%). The best allowlisted map, `304be_expertplus`, scores 33.3% but is a training song. Options: lead with 304be labelled as a training song and show 2dd26 as "never heard"; or allowlist another held-out CC song that suits the fly (Person 2).

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
- [ ] **C6. Looming lesion measurably lowers hit rate** (Person 1, or Person 2's measurement session). Acceptance test; not re-measured since #36 changed how lesions act. Run Person 1's `play_conditions` eval on `real`, untouched vs. looming detectors lesioned.
- [ ] **C2. Versus UI** (Person 3). The client sends `brains: [a, b]`, handles `slot` 1 in frames, events and `ended`, and the screen splits. Today `session-client.ts` fails unless there is exactly one slot.
- [ ] **C3. Live neuron activity.** Pin the binary layout in PROTOCOL.md section 6 (Person 1); stream kinds 1 and 2 (Person 2); decode them and light the neuron cloud (Person 3). Neuron order matches `neurons.json` and `ready.cell_types`. Must hold 30 Hz on the public host (E2).
- [ ] **C4. Replay fallback** (Person 3, Person 2 pairs). `scripts/record_replays.py` records each demo map's frames and events for `real` and the Versus control; a replay player in the front end runs them with the server off. Both are stubs today.
- [ ] **C5. Keep the live song and brain picker** (Person 3). Open PR #44 deletes `live-songs.tsx`, the only UI that starts a fly session, plus `live-session.spec.ts` and `live-integration.spec.ts`. Merged as is, the browser could only play BeatSaver previews, which the fly never plays. Keep the picker, or move it, before #44 merges.

## Milestone D: public URL

- [x] **D1. Hosting decision.** The front end is at https://flysaber.jackie-courtney.com, a Cloudflare Worker (#41) that proxies `/api/*` and `/ws/*` to its `BACKEND_URL` secret, so the browser stays same-origin and the server needs no CORS. Person 1 runs the backend on Vultr with `scripts/deploy_vultr.sh` (#42).
- [ ] **D3. Backend hostname and TLS** (Person 1). `docs/CI_CD.md` sets `BACKEND_URL=http://<server ip>:8000`; check that the Worker can fetch a bare IP and port, and if it can't, give the host a DNS name. Use a single-level name such as `flysaber-api.jackie-courtney.com`: Cloudflare's free certificate doesn't cover `api.flysaber.jackie-courtney.com`.
- [ ] **D5. Allowlisted audio only on the public URL** (Person 3, spec 8.4). The in-browser BeatSaver search plays any map's audio, with no production gate. Hide it, or limit it to the allowlist, on the public build. #44 makes it the sidebar's default view.
- [ ] **D6. Size the public host** (Person 1). The real brain runs about 2.1x real time on a laptop CPU, so a Versus session (2 brain slots) takes most of a laptop. Set `MAX_SESSIONS` to what the Vultr host keeps at 30 Hz with activity streaming.
- [ ] **D7. Fix the "Workers Builds: flysaber" check** (whoever set up Cloudflare). It fails on every PR.

## Milestone E: ship

- [ ] **E1. README results with real numbers** (Person 3 owns `docs/RESULTS.md`): held-out hit rate, real vs. scrambled, the looming lesion, the blindfold.
- [ ] **E2. Acceptance tests** (handoff section 18) pass on the public host, including 30 Hz with activity streaming and replay with the server off.
- [ ] **E3. Rehearse the demo** (all), once live and once from replays.

## Risks to check

- The held-out song plays to the fly's weak spots. Held-out hit rate by note: bottom row middle lanes 41 to 49%, outer lanes 17 to 27%, middle row centre 0%; down cuts 46%, up 22%, left and right 2 to 4%. `2dd26_expert` alternates up and down cuts and puts 18% of its notes in the centre cells.
- The sim runs notes at 7.0 m/s while the preview uses each map's own speed. Hit timing matches (both use `time_sec`); only note spacing on screen differs.

## Can wait

- **B4. Improve the trained fly.** Not approved. If it is: target up and lateral cuts and off-centre cells, for example by reweighting those classes in BC; longer clips won't help (whole songs and 5 s clips score the same).
- Fly's-eye view (eye stream kind 3 and `/api/eye_lattice`), opto and the remaining moods in the UI, Brain Explorer live traces (`/api/explorer`), Gemini narrator (first to cut), You vs. the Fly, curriculum and DAgger, PPO, matched MLP baseline, Fly Runner.
