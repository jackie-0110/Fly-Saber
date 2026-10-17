"""Synthetic eye stimuli for checks and benchmarks.

Owner: Person 1.
"""

import numpy as np
import pandas as pd


def looming_eye(n_frames: int, columns: pd.DataFrame) -> np.ndarray:
    """A dark disc expanding on a bright background in the left eye, centered on its columns.

    Returns (frames, 2, columns): luminance, and a neutral 0.5 color channel.
    """
    left = (columns["side"] == "L").to_numpy()
    h = columns[["hex1", "hex2"]].to_numpy(dtype=float)
    center = h[left].mean(axis=0)
    dist = np.linalg.norm(h - center, axis=1)
    radius = np.linspace(0, dist[left].max(), n_frames)[:, None]
    luminance = np.where(left & (dist < radius), 0.0, 1.0)
    return np.stack([luminance, np.full_like(luminance, 0.5)], axis=1).astype(np.float32)
