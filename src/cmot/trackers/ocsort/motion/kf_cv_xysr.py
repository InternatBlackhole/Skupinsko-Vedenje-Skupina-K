"""Constant-velocity Kalman filter in (cx, cy, area, aspect) space.

The paper's default motion model (inherited from SORT), reproduced exactly as
in the official ``KalmanBoxTracker``:

    state       x = [cx, cy, s, r, vcx, vcy, vs]     s = w*h (area), r = w/h
    measurement z = [cx, cy, s, r]

The aspect ratio r has no velocity term. Noise matrices start from identity and
are scaled by the multipliers below, applied in the same order as the official
code so the matrices are bit-identical.
"""
import numpy as np

from cmot.trackers.ocsort.motion.base import MotionModel
from cmot.trackers.ocsort.motion.kalman import KalmanFilter

_F = np.array(
    [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)
_H = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.float64,
)


def box_to_z(box: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, s, r] (official ``convert_bbox_to_z``)."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = box[0] + w / 2.0
    y = box[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def x_to_box(x: np.ndarray) -> np.ndarray:
    """State -> [x1, y1, x2, y2] (official ``convert_x_to_bbox``)."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape(4)


class KFConstantVelocityXYSR(MotionModel):
    def __init__(
        self,
        box_xyxy: np.ndarray,
        r_scale_sr: float = 10.0,
        p_scale_velocity: float = 1000.0,
        p_scale: float = 10.0,
        q_scale_vs: float = 0.01,
        q_scale_velocity: float = 0.01,
    ):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = _F
        kf.H = _H
        kf.R[2:, 2:] *= r_scale_sr          # area / aspect measurements are noisier
        kf.P[4:, 4:] *= p_scale_velocity    # velocities are unobserved at birth
        kf.P *= p_scale
        kf.Q[-1, -1] *= q_scale_vs
        kf.Q[4:, 4:] *= q_scale_velocity
        kf.x[:4] = box_to_z(box_xyxy)
        self.kf = kf

    def box_to_measurement(self, box_xyxy):
        return box_to_z(box_xyxy)

    def measurement_to_cxcywh(self, z):
        cx, cy, s, r = np.asarray(z).flatten()
        return cx, cy, np.sqrt(s * r), np.sqrt(s / r)

    def cxcywh_to_measurement(self, cx, cy, w, h):
        return np.array([cx, cy, w * h, w / float(h)]).reshape((4, 1))

    def predict(self, control=None, guard=True):
        # Keep the predicted area non-negative (official KalmanBoxTracker.predict).
        if guard and (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict(control)

    def update(self, z):
        self.kf.update(z)

    def box(self):
        return x_to_box(self.kf.x)

    def velocity(self):
        return self.kf.x[4:6, 0].copy()

    def snapshot(self):
        return self.kf.snapshot()

    def restore(self, snap):
        self.kf.restore(snap)
