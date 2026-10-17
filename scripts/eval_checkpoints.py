"""Let trained flies play: held-out songs, and every lesion and mood from the UI.

Owner: Person 1. Run on HiPerGator (GPU) from the repo root:

    # Compare brains on held-out and training songs, untouched:
    uv run python scripts/eval_checkpoints.py checkpoints/*/*/seed*/bc_1000.npz
    # Every UI control on one fly (the demo check):
    uv run python scripts/eval_checkpoints.py checkpoints/real/fast/seed0/bc_950.npz --full

Writes eval_<checkpoint>.json next to each checkpoint and prints markdown tables.
All brains and conditions play the same song windows, so differences are paired.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from brain.model.controls import load_controls
from training.play_eval import conditions, load_trained, play_conditions


def label(path: Path) -> str:
    """checkpoints/<variant>/<run>/seed<n>/bc_<k>.npz -> variant/run/seed<n>/bc_<k>."""
    parts = path.with_suffix("").parts
    return (
        "/".join(parts[parts.index("checkpoints") + 1 :]) if "checkpoints" in parts else str(path)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--full", action="store_true", help="every lesion, mood, and noise setting")
    ap.add_argument("--splits", nargs="+", default=["heldout", "train"])
    ap.add_argument("--windows", type=int, default=256, help="song windows per split")
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument(
        "--fixed-baseline",
        action="store_true",
        help="keep the untouched brain's readout baseline under every control (old behavior)",
    )
    ap.add_argument("--subgraph", type=Path, help="subgraph dir, e.g. a bundle's subgraph/")
    ap.add_argument("--maps-dir", help="override the training config's maps_dir")
    args = ap.parse_args()

    controls = load_controls()
    named = conditions(controls, args.full)
    summary: dict[str, dict[str, Any]] = {}
    for path in args.checkpoints:
        print(f"evaluating {label(path)} ...", flush=True)
        trained = load_trained(path, args.subgraph)
        if args.maps_dir:
            cfg = dataclasses.replace(trained.train_cfg, maps_dir=args.maps_dir)
            trained = dataclasses.replace(trained, train_cfg=cfg)
        result: dict[str, Any] = {"checkpoint": str(path), "meta": trained.meta["metrics"]}
        for split in args.splits:
            result[split] = play_conditions(
                trained,
                controls,
                named,
                split,
                args.windows,
                args.window_sec,
                rebaseline=not args.fixed_baseline,
            )
        out = path.with_name(f"eval_{path.stem}.json")
        out.write_text(json.dumps(result, indent=1))
        summary[label(path)] = result

    print("\n## Untouched brain, greedy play (hit rate)\n")
    print("| checkpoint | " + " | ".join(args.splits) + " |")
    print("|---|" + "---|" * len(args.splits))
    for name, result in summary.items():
        rates = [f"{result[s]['untouched']['hit_rate']:.1%}" for s in args.splits]
        print(f"| {name} | " + " | ".join(rates) + " |")

    if args.full:
        split = args.splits[0]
        for name, result in summary.items():
            base = result[split]["untouched"]["hit_rate"]
            print(f"\n## {name}: every control, {split} songs\n")
            print("| condition | hit rate | change | bad cuts | bombs |")
            print("|---|---|---|---|---|")
            for cond, r in result[split].items():
                change = "" if cond == "untouched" else f"{r['hit_rate'] - base:+.1%}"
                print(
                    f"| {cond} | {r['hit_rate']:.1%} | {change} | {r['bad_cuts']} | {r['bombs']} |"
                )


if __name__ == "__main__":
    main()
