"""Download MaleCNS v1.0 bulk files into data/raw (about 1.1 GB).

Owner: Person 1. Usage: uv run python scripts/data/fetch_data.py
"""

import urllib.request
from pathlib import Path

from brain.connectome.load_malecns import BASE_URL, FILES

RAW_DIR = Path("data/raw/malecns")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"downloading {name} ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        tmp.rename(dest)


if __name__ == "__main__":
    main()
