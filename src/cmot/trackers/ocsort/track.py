"""A single OC-SORT track (official ``KalmanBoxTracker`` with a pluggable motion model)."""
from typing import Callable

import numpy as np

from cmot.trackers.ocsort.motion.base import MotionModel
from cmot.trackers.ocsort.observations import NO_OBSERVATION, speed_direction
from cmot.trackers.ocsort.oru import ObservationCentricReUpdate


class OCSortTrack:
    def __init__(
        self,
        det: np.ndarray,
        track_id: int,
        motion_factory: Callable[[np.ndarray], MotionModel],
        delta_t: int = 3,
        use_oru: bool = True,
        oru_trajectory: str = "linear",
    ):
        self.id = track_id
        self.motion = motion_factory(det[:4])
        self.oru = ObservationCentricReUpdate(self.motion, enabled=use_oru, trajectory=oru_trajectory)
        self.delta_t = delta_t

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # As in the official code, the birth detection initialises the state but is
        # not stored as an observation.
        self.last_observation = NO_OBSERVATION
        self.observations: dict[int, np.ndarray] = {}  # age -> [x1, y1, x2, y2, score]
        self.velocity: np.ndarray | None = None  # observed direction (dy, dx) used by OCM

    def predict(self, control: np.ndarray | None = None) -> np.ndarray:
        self.motion.predict(control)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.motion.box()

    def update(self, det: np.ndarray | None) -> None:
        if det is None:
            self.oru.update(None)
            return

        if self.last_observation.sum() >= 0:  # has a previous observation
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.velocity = speed_direction(previous_box, det)

        self.last_observation = det
        self.observations[self.age] = det
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.oru.update(self.motion.box_to_measurement(det))

    def state_box(self) -> np.ndarray:
        return self.motion.box()
