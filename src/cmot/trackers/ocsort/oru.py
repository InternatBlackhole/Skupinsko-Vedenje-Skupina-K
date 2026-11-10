"""Observation-centric Re-Update (ORU).

When a track is re-associated after being lost, OC-SORT rolls its motion model
back to the moment it was lost and replays the gap with a *virtual trajectory*
between the last observation before the gap and the new one. This removes the
error the filter accumulated while it was only predicting.

Behaviour mirrors the official ``KalmanFilterNew.freeze/unfreeze``:

* the state is saved after the predict of the first missed frame;
* on re-observation the state is restored and, for each gap frame, the filter
  is updated with the virtual measurement and then predicted (bare filter
  predict, no guard), except after the last one;
* the last virtual measurement equals the new observation, and the new
  observation is then applied once more by the regular update.

The virtual trajectory is pluggable (``TRAJECTORIES``). The paper uses linear
interpolation; a collective-motion trajectory can be added here later.
"""
from typing import Callable

import numpy as np

from cmot.trackers.ocsort.motion.base import MotionModel

CXCYWH = tuple[float, float, float, float]


def linear_trajectory(start: CXCYWH, end: CXCYWH, gap: int) -> list[CXCYWH]:
    """``gap`` evenly spaced boxes from just after ``start`` up to and including ``end``."""
    x1, y1, w1, h1 = start
    x2, y2, w2, h2 = end
    dx = (x2 - x1) / gap
    dy = (y2 - y1) / gap
    dw = (w2 - w1) / gap
    dh = (h2 - h1) / gap
    return [
        (x1 + (i + 1) * dx, y1 + (i + 1) * dy, w1 + (i + 1) * dw, h1 + (i + 1) * dh)
        for i in range(gap)
    ]


TRAJECTORIES: dict[str, Callable[[CXCYWH, CXCYWH, int], list[CXCYWH]]] = {
    "linear": linear_trajectory,
}


class ObservationCentricReUpdate:
    """Wraps a track's motion model and routes every per-frame update through ORU."""

    def __init__(self, motion: MotionModel, enabled: bool = True, trajectory: str = "linear"):
        self.motion = motion
        self.enabled = enabled
        self.trajectory = TRAJECTORIES[trajectory]
        self.history: list[np.ndarray | None] = []  # one measurement (or None) per update call
        self.observed = False
        self._saved = None  # (motion snapshot, history) taken when the track got lost

    def update(self, z: np.ndarray | None) -> None:
        self.history.append(z)
        if z is None:
            if self.observed and self.enabled:
                self._saved = (self.motion.snapshot(), list(self.history))
            self.observed = False
            self.motion.update(None)
            return
        if not self.observed:
            self._re_update()
        self.observed = True
        self.motion.update(z)

    def _re_update(self) -> None:
        if self._saved is None:
            return
        new_history = self.history
        snap, saved_history = self._saved
        self._saved = None
        self.motion.restore(snap)
        self.history = saved_history[:-1]

        observed_at = [i for i, z in enumerate(new_history) if z is not None]
        index1, index2 = observed_at[-2], observed_at[-1]
        gap = index2 - index1
        start = self.motion.measurement_to_cxcywh(new_history[index1])
        end = self.motion.measurement_to_cxcywh(new_history[index2])

        for i, (cx, cy, w, h) in enumerate(self.trajectory(start, end, gap)):
            virtual = self.motion.cxcywh_to_measurement(cx, cy, w, h)
            self.history.append(virtual)
            self.motion.update(virtual)
            if i != gap - 1:
                self.motion.predict(guard=False)
