"""Learning curves for the BC runs: hit rate and position accuracy over training.

Owner: Person 1. Usage (repo root, after copying or on HiPerGator):
    uv run python scripts/plot_learning_curves.py [--checkpoints checkpoints] [--out docs/figures]

Reads checkpoints/<variant>/<run>/seed<n>/bc_log.json. Each brain is one color; seed 0 is a
solid line, seed 1 dashed (seeds can stop at different updates). Writes
learning_curves.png, learning_curves.svg, and learning_curves.md (the same numbers as a table).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots 1 to 4 (dataviz reference palette, light mode). Slots 3 and 4
# are below 3:1 contrast on the surface, so every line is also direct-labeled.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# (variant, run_name) -> (legend label, short direct label), in fixed color order.
RUNS = {
    ("real", "fast"): ("Real wiring, brain trained", "Real, trained"),
    ("real", "readout_only"): ("Real wiring, readout only", "Real, readout only"),
    ("rewired_layered", "fast"): ("Partners shuffled within layers", "Shuffled partners"),
    ("sign_shuffled", "fast"): ("Excitation/inhibition shuffled", "Shuffled signs"),
}
SEED_STYLES = ["-", "--", ":"]


def load_runs(root: Path) -> dict[tuple[str, str], list[list[dict[str, float]]]]:
    """(variant, run) -> one log (list of rows) per seed, in seed order."""
    runs: dict[tuple[str, str], list[list[dict[str, float]]]] = defaultdict(list)
    for f in sorted(root.glob("*/*/seed*/bc_log.json")):
        runs[(f.parts[-4], f.parts[-3])].append(json.loads(f.read_text()))
    return runs


def series(log: list[dict[str, float]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Updates and values of one run; "position" averages the red and blue position heads."""
    x = np.array([r["update"] for r in log])
    if metric == "position":
        y = np.array([(r["head_accuracy"][0] + r["head_accuracy"][2]) / 2 for r in log])  # type: ignore[index]
    else:
        y = np.array([r[metric] for r in log])
    return x, y


def plot(runs: dict[tuple[str, str], list[list[dict[str, float]]]], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
    panels = [("hit_rate", "Notes hit in free play"), ("position", "Saber cell matches the expert")]
    rows = []
    for ax, (metric, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ends = []
        for color, (key, (name, short)) in zip(COLORS, RUNS.items(), strict=True):
            for seed, log in enumerate(runs.get(key, [])):
                x, y = series(log, metric)
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2,
                    ls=SEED_STYLES[seed % len(SEED_STYLES)],
                    label=name if seed == 0 else None,
                )
                if seed == 0:
                    ends.append((y[-1], x[-1], short, color))
                if metric == "hit_rate":
                    best = max(log, key=lambda r: r["hit_rate"])
                    rows.append(
                        (name, seed, int(x[-1]), y[-1], best["hit_rate"], int(best["update"]))
                    )
        last = max(int(xe) for _, xe, _, _ in ends) if ends else 1000
        ax.set_xlim(0, last * 1.28)  # room on the right for direct labels
        ax.set_xticks(
            [t for t in matplotlib.ticker.MaxNLocator(5).tick_values(0, last) if t <= last]
        )
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ymin, ymax = ax.get_ylim()
        gap = 0.07 * (ymax - ymin)
        placed: list[float] = []
        for y, xe, short, color in sorted(ends):
            y_text = max([y] + [p + gap for p in placed])  # nudge labels apart
            placed.append(y_text)
            ax.plot([xe + 15, xe + 35], [y, y_text], color=color, lw=2, solid_capstyle="round")
            ax.annotate(short, (xe + 45, y_text), va="center", fontsize=9.5, color=INK_2)
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
        ax.set_xlabel("Training update", color=INK_2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color=INK_2, lw=2, ls="--"))
    labels.append("dashed: second seed")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Behavior cloning on Fly Saber: the real fly brain vs control brains",
        x=0.012,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    png = out / "learning_curves.png"
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(out / "learning_curves.svg", facecolor=SURFACE)
    plt.close(fig)

    table = [
        "| Brain | Seed | Updates | Final hit rate | Best hit rate (update) |",
        "|---|---|---|---|---|",
    ]
    for name, seed, upd, final, top, at in rows:
        table.append(f"| {name} | {seed} | {upd} | {final:.1%} | {top:.1%} ({at}) |")
    (out / "learning_curves.md").write_text("\n".join(table) + "\n")
    return png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    runs = load_runs(args.checkpoints)
    if not runs:
        raise SystemExit(f"no bc_log.json under {args.checkpoints}/<variant>/<run>/seed*/")
    print(f"wrote {plot(runs, args.out)}")
    print((args.out / "learning_curves.md").read_text())


if __name__ == "__main__":
    main()
