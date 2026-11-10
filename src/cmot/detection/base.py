"""Detection sources.

A source returns, per sequence, ``{frame: (N, 5) [x1, y1, x2, y2, score]}``.
Detections are produced once (cached) and then re-used by every tracker and
motion-model variant, so differences between runs come from tracking alone.
"""
from abc import ABC, abstractmethod

import numpy as np

from cmot.datasets.base import Sequence

EMPTY = np.empty((0, 5))


class DetectionSource(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """Short identifier recorded with every run (e.g. ``gt_oracle`` or a weights run id)."""

    @abstractmethod
    def load(self, seq: Sequence) -> dict[int, np.ndarray]:
        """Detections for every frame of ``seq`` that has any."""


def group_by_frame(rows: np.ndarray) -> dict[int, np.ndarray]:
    """``(K, 6) [frame, x1, y1, x2, y2, score]`` -> ``{frame: (N, 5)}``."""
    out: dict[int, np.ndarray] = {}
    if len(rows) == 0:
        return out
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    frames, starts = np.unique(rows[:, 0].astype(int), return_index=True)
    for f, a, b in zip(frames, starts, list(starts[1:]) + [len(rows)]):
        out[int(f)] = rows[a:b, 1:6].copy()
    return out
