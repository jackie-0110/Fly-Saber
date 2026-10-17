"""Generate a deterministic SVG illustration, not connectome or activity data.

Run from any directory with Python 3. No third-party dependencies are needed.
The browser loads the output as one static image and runs no simulation.
"""

import math
import random
from pathlib import Path


def main() -> None:
    rng = random.Random(42)
    points: list[str] = []
    # Stylized bilateral brain lobes and cord, deliberately not anatomical data.
    lobes = [
        (99, 150, 71, 64, 900),
        (321, 150, 71, 64, 900),
        (166, 123, 60, 64, 750),
        (254, 123, 60, 64, 750),
        (210, 169, 80, 34, 480),
        (195, 289, 31, 39, 280),
        (225, 289, 31, 39, 280),
        (194, 353, 30, 43, 300),
        (226, 353, 30, 43, 300),
        (195, 420, 26, 38, 250),
        (225, 420, 26, 38, 250),
        (210, 455, 24, 23, 150),
    ]
    for cx, cy, rx, ry, count in lobes:
        for _ in range(count):
            angle = rng.random() * math.tau
            radial = math.sqrt(rng.random())
            x = cx + math.cos(angle) * rx * radial
            y = cy + math.sin(angle) * ry * radial
            depth = rng.random()
            hotspot = ((x - 150) / 28) ** 2 + ((y - 175) / 21) ** 2 < 1 or ((x - 270) / 28) ** 2 + (
                (y - 175) / 21
            ) ** 2 < 1
            if hotspot and depth > 0.28:
                color = "#f0dfae"
            elif depth > 0.82:
                color = "#bfd7d6"
            else:
                color = "#819da9"
            opacity = rng.uniform(0.18, 0.65) if not hotspot else rng.uniform(0.4, 0.95)
            radius = rng.uniform(0.45, 1.2) if not hotspot else rng.uniform(0.55, 1.4)
            points.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.2f}" '
                f'fill="{color}" opacity="{opacity:.2f}"/>'
            )
    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="510" '
            'viewBox="0 0 420 510">',
            "<title>Illustrative neural activity preview</title>",
            "<desc>Seeded decorative point cloud. Not real neuron coordinates, "
            "biological anatomy, or measured firing rates.</desc>",
            *points,
            "</svg>",
            "",
        ]
    )
    target = Path(__file__).resolve().parents[1] / "public/previews/neural-preview.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")
    print(f"Wrote static illustration: {target.name}")


if __name__ == "__main__":
    main()
