"""Linear Kalman filter.

Numerically identical to the ``KalmanFilterNew`` predict/update used by the
official OC-SORT (a filterpy derivative): same operation order, Joseph-form
covariance update and ``np.linalg.inv`` for the innovation covariance. The ORU
freeze/unfreeze logic that the official code keeps inside the filter lives in
``ocsort/oru.py`` instead, so this class stays a plain filter.
"""
import numpy as np


class KalmanFilter:
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.B: np.ndarray | None = None
        self._I = np.eye(dim_x)

    def predict(self, u: np.ndarray | None = None) -> None:
        if self.B is not None and u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        self.P = (self.F @ self.P) @ self.F.T + self.Q

    def update(self, z: np.ndarray | None) -> None:
        if z is None:
            return
        z = np.asarray(z, dtype=np.float64).reshape(self.dim_z, 1)
        H, R = self.H, self.R
        y = z - H @ self.x
        PHT = self.P @ H.T
        S = H @ PHT + R
        K = PHT @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = self._I - K @ H
        self.P = (I_KH @ self.P) @ I_KH.T + (K @ R) @ K.T

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.x, self.P = snap[0].copy(), snap[1].copy()
