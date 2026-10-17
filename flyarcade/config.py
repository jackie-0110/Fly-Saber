"""Load YAML configs into frozen dataclasses.

Owner: All. See docs/ARCHITECTURE.md conventions.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

T = TypeVar("T")


def load_config(cls: type[T], name: str, **overrides: Any) -> T:
    """Build `cls` from `configs/{name}.yaml`; unknown or missing keys raise TypeError."""
    values = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()) or {}
    return cls(**{**values, **overrides})
