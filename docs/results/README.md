# Results (Person 1)

The headline numbers with sources, the lesion matrix, mood validation and live control
measurements are summarised in [SUMMARY.md](SUMMARY.md).

Behavior cloning runs on HiPerGator (one NVIDIA L4 each). One folder per run:
`bc/<variant>/<run_name>/seed<n>/` with `bc_log.json` (a row every 50 updates),
`config.json`, and `eval_bc_<update>.json` (greedy play on held-out and training songs,
256 windows of 10 s, same windows for every brain; `scripts/training/eval_checkpoints.py`).

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

The demo fly is `data/bundles/real/` (real wiring, brain trained, seed 0, update 950),
shipped with the exact subgraph it was trained on.

## UI controls on the demo fly

What each lesion, mood and preset does to `data/bundles/real/bc_950.npz`, so the live controls
change play visibly without breaking it. Held-out songs, 64 windows of 10 s, the same
windows for every condition, readout re-centered for each condition as the server does
(#36). Differences of 2 to 3 points are within noise.

- `controls/eval_bc_950_full.json`: every lesion and every mood at -1 and +1
  (`scripts/training/eval_checkpoints.py --full --splits heldout --windows 64`).
- `controls/moods_bc_950.json`: each mood at -0.5, -0.3, -0.15, +0.15, +0.3, +0.5, and four
  presets at 0.15 and 0.3 (`training.play_eval.play_conditions`).

| Condition | Hit rate | Bad cuts | Bombs |
|---|---|---|---|
| Untouched | 32.9% | 50 | 14 |
| Lesion looming detectors | 14.1% | 114 | 4 |
| Lesion left eye / right eye | 7.1% / 7.6% | 15 / 79 | 18 / 26 |
| Lesion motion detectors (T4, T5) | 2.1% | 63 | 5 |
| Arousal -0.3 / -0.15 / +0.15 / +0.3 / +0.5 | 6.0 / 14.5 / 33.8 / 23.2 / 8.8% | 62 / 73 / 79 / 124 / 306 | |
| Reward drive -0.5 / +0.15 / +0.5 | 29.1 / 17.2 / 16.7% | 108 / 47 / 92 | 16 / 24 / 20 |
| Calm -0.5 / -0.15 / +0.15 / +0.5 | 18.2 / 33.2 / 27.6 / 14.8% | 68 / 47 / 91 / 65 | |
| Depressed (arousal -0.3, reward drive -0.3) | 8.9% | 65 | 12 |
| Anxious (arousal +0.3, calm -0.3) | 10.5% | 176 | 7 |
| Excited (arousal +0.3, reward drive +0.3) | 9.9% | 259 | 9 |
| Content (calm +0.3, reward drive +0.3) | 20.9% | 56 | 25 |

Read honestly: the looming lesion more than halves the hit rate, but it is one of the
milder visual lesions; blinding either eye or removing the motion detectors drops the fly
to about chance. Arousal is steep toward drowsy, so its slider stops at -0.15; lowering
reward drive barely matters, so that slider starts at 0. The presets are combinations of
the three neuromodulators, not emotions the connectome encodes: depressed stops swinging,
while anxious and excited swing at everything and cut the wrong way. Slider ranges and
presets live in `configs/controls.yaml`.
