"""Scrape the charts-only BeatSaver training set and write its index (spec 8.1).

Owner: Person 2. See docs/project/FLY_SABER_HANDOFF.md for the spec.
"""

import json
from collections import Counter
from pathlib import Path

from fly_saber.config import load_config
from fly_saber.maps.allowlist import load_allowlist
from fly_saber.maps.beatsaver_client import make_client
from fly_saber.maps.training_set import TrainingMapsConfig, build_training_set


def main() -> None:
    cfg = load_config(TrainingMapsConfig, "training_maps")
    with make_client(cfg.request_interval_sec) as client:
        index, skipped = build_training_set(client, cfg, load_allowlist())
    index.sort(key=lambda row: row["map_id"])
    (Path(cfg.out_dir) / "index.json").write_text(json.dumps(index, indent=1))

    songs = {row["key"]: row["split"] for row in index}
    print(f"{len(songs)} songs, {len(index)} difficulties")
    print("songs per split:", dict(Counter(songs.values())))
    print("difficulties per split:", dict(Counter(row["split"] for row in index)))
    print("skipped:")
    for reason, count in skipped.most_common():
        print(f"  {count:4d}  {reason}")


if __name__ == "__main__":
    main()
