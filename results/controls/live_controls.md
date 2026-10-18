# Live controls on the served brain

Measured 2026-10-18 against the running server (`bundles/real`, real wiring, update 950),
through the same `set_interventions` message the page sends. One condition per session:
start, apply the control, wait 3 s for the server to re-centre the readout, then play
`304be_expertplus` from 20 s to 35 s in real time and count `event` messages. Hit rate =
hits / (hits + misses + bad cuts). One 15 s window of about 55 notes per condition, so a
difference of a few points is noise; these numbers set slider ranges, they are not an eval.
Script: `frontend/scripts/measure-live-controls.mjs`.

The held-out eval in `eval_bc_950_full.json` (64 windows of 10 s) is the reference for
lesions and moods; this file covers what it does not: stimulation and the eye controls.

## Lesions and stimulation

| Condition | Hits / judged | Hit rate | Bad cuts |
|---|---|---|---|
| Untouched | 19 / 55 | 34.5% | 0 |
| Lesion motion detectors (T4, T5) | 3 / 54 | 5.6% | 1 |
| Lesion looming detectors | 7 / 54 | 13.0% | 8 |
| Stimulate giant fiber at 5.0 (Trigger escape) | 14 / 53 | 26.4% | 0 |

The two lesions agree with the held-out eval (2.1% and 14.1%). Driving the escape command
neuron at full strength costs the fly about a quarter of its hits, a visible but modest
change: in this model the giant fiber is one of 1,310 descending neurons feeding the
saber readout, not a jump the game can show.

## Eye controls

| Control | Level | Hit rate |
|---|---|---|
| Blur (extra acceptance FWHM) | 2° / 4° / 6° / 10° / 20° | 20.4 / 13.0 / 5.7 / 0.0 / 3.8% |
| Facets lost | 10% / 20% / 30% / 50% / 100% | 11.1 / 0.0 / 0.0 / 0.0 / 0.0% |
| Visual noise (luminance std) | 0.02 / 0.05 / 0.1 / 0.15 / 0.3 | 18.5 / 12.7 / 7.3 / 3.6 / 1.9% |

The trained fly is very sensitive to degraded vision, so the page's sliders end where it
stops playing: blur 0 to 8°, facets lost 0 to 25%, visual noise 0 to 0.15. The whole
travel of each slider then shows a gradient instead of an instant collapse.

## Neural noise

| Noise std | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| Hit rate | 1.9% | 1.8% | 1.8% | 0.0% | 1.9% | 1.8% | 1.8% | 0.0% |

Any neural noise, down to 1/250 of `max_noise`, takes the fly from 34.5% to about 2% with
about 20 bad cuts per window, so the page offers it as an on/off switch. The likely cause
is in `src/brain/model/brain.py`: noise is added to every neuron each substep, and the readout
(`readout_norm`) divides descending-neuron rates by their resting spread (`dn_std`), which
is tiny, so the noise is amplified with the signal. Worth a look by the model owner if
noise should behave like a dial.
