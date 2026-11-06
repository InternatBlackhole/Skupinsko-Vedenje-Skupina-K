"""Interface shared by all trackers in ``cmot.trackers``.

Each tracker lives in its own sub-package (``ocsort/``, ``bytetrack/``, ...) with
its own association code and motion models; only this interface is shared, so
the runner and evaluation can drive any tracker the same way.
"""
from abc import ABC, abstractmethod

import numpy as np


class Tracker(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict) -> "Tracker":
        """Build the tracker from its resolved config dict."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks and restart ids; called at the start of every sequence."""

    @abstractmethod
    def update(self, dets: np.ndarray) -> np.ndarray:
        """Process one frame.

        Args:
            dets: ``(N, 5)`` detections ``[x1, y1, x2, y2, score]`` in image pixels.
        Returns:
            ``(M, 5)`` reported tracks ``[x1, y1, x2, y2, track_id]`` (ids start at 1).
        """
