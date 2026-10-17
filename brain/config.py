"""Load YAML configs into frozen dataclasses.

Owner: Person 1.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str | Path, cls: type[T]) -> T:
    """Read a YAML file and build `cls` from it, rejecting unknown keys."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class SubgraphConfig:
    dataset: str
    raw_dir: str
    out_dir: str
    min_synapses: int
    column_stride: int
    columnar_types: list[str]
    projection_types: list[str]
    descending_superclass: str
    max_intermediates: int
    max_neurons: int
    hex_inherit_superclasses: list[str]
    hex_inherit_passes: int
