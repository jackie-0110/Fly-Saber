# Results (Person 1)

Behavior cloning runs on HiPerGator (one NVIDIA L4 each). One folder per run:
`bc/<variant>/<run_name>/seed<n>/` with `bc_log.json` (a row every 50 updates),
`config.json`, and `eval_bc_<update>.json` (greedy play on held-out and training songs,
256 windows of 10 s, same windows for every brain; `scripts/eval_checkpoints.py`).

Settings: 5 s windows, 256 games per update, readout fitted on expert play before training,
brain learning rate 1e-4 (readout-only runs: 0, brain frozen), notes at a fixed 7 m/s.
Seed 1 runs stopped at 600 updates. `bc/real/seed0/` is the first run (64 games, 10 s
windows, brain learning rate 1e-5, 500 updates).

| Brain | Seed | Held-out songs | Training songs |
|---|---|---|---|
| Real wiring, brain trained | 0 / 1 | 31.4% / 29.9% | 28.4% / 26.8% |
| Real wiring, readout only | 0 / 1 | 33.9% / 31.7% | 29.9% / 28.3% |
| Partners shuffled within layers | 0 / 1 | 51.8% / 44.4% | 46.2% / 39.9% |
| Excitation/inhibition shuffled | 0 / 1 | 20.0% / 17.0% | 17.9% / 16.0% |

Hit rate = notes hit / notes judged, greedy play, no expert. Learning curves:
`docs/figures/learning_curves.png` (table: `learning_curves.md`).

Read honestly: every brain plays unseen songs at least as well as training songs.
Shuffling excitation and inhibition roughly halves performance. Shuffling partners within
layers *improves* it; the likely reason is that the real circuit compresses vision into
fly-relevant signals (for example LPLC2 reports looming wherever it happens) and discards
where-on-screen detail this game needs. Training the brain's own parameters did not beat
training the readout alone in free play.

The demo fly is `bundles/real_fast/` (real wiring, brain trained, seed 0, update 950),
shipped with the exact subgraph it was trained on.
