"""Observation-history helpers used by OCM (observation-centric momentum).

Observations are boxes ``[x1, y1, x2, y2, score]``. A track that has no
observation yet uses the placeholder ``[-1, -1, -1, -1, -1]`` (kept from the
official code so the k-previous observations of all tracks fit in one array).
"""
import numpy as np

NO_OBSERVATION = np.array([-1, -1, -1, -1, -1])


def k_previous_obs(observations: dict[int, np.ndarray], cur_age: int, k: int) -> np.ndarray:
    """Observation ``k`` steps back; else the closest more recent one; else the latest."""
    if len(observations) == 0:
        return NO_OBSERVATION
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Unit vector ``(dy, dx)`` from the centre of ``bbox1`` to the centre of ``bbox2``."""
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_batch(dets: np.ndarray, tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions from every track's previous observation to every detection.

    Returns ``(dy, dx)``, each of shape ``(num_tracks, num_dets)``.
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm
