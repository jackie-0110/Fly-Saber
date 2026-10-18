"""Load YAML configs into frozen dataclasses.

Owner: Person 1.
"""

from dataclasses import dataclass, field, fields
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
    modulatory_nts: list[str] = field(default_factory=list)
    max_modulatory_per_nt: int = 0
    min_modulatory_synapses: int = 0


@dataclass(frozen=True)
class ModelConfig:
    subgraph_dir: str
    frame_ms: float
    substeps: int
    tau_init_ms: float
    tau_min_ms: float
    bias_init: float
    activation: str
    in_weight_scale: float
    input_scale: float
    action_heads: list[int]
    # Neuromodulation: log-gain on each target per unit change of its modulators' rate
    # from rest. Empty disables it. See src/brain/model/brain.py.
    modulation_gain: dict[str, float] = field(default_factory=dict)
    modulation_max_log_gain: float = 1.0
    rate_max: float = 10.0  # saturation level for activation relu_sat
    remat: bool = True  # gradient checkpointing per brain step; trades compute for memory
    # Readout normalization: actions read (DN rate - mean) / std, both measured once on
    # flickering visual input when params are created. See src/brain/model/brain.py.
    readout_norm: bool = True
    calibration_stimuli: int = 16
    calibration_frames: int = 90
    readout_std_floor: float = 0.1  # fraction of the median DN std; avoids dividing by ~0
    readout_clip: float = 5.0  # normalized DN values are clipped to +-this
    freeze_brain: bool = False  # stop gradients at the DNs: train only the readout, much faster
    # Live baseline: the readout subtracts the DN rates of the brain at rest under the
    # *current* params (rest_dn), recomputed each update, so brain drift cancels out.
    live_baseline: bool = True
    baseline_frames: int = 45
