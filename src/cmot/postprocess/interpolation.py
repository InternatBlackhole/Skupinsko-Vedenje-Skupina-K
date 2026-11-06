"""Linear interpolation of short track gaps (official ``tools/interpolation.py``, ``dti``).

Offline post-processing used by OC-SORT for MOT17/MOT20 test submissions only;
not part of the online tracker.
"""
import numpy as np


def interpolate_tracks(rows: np.ndarray, n_min: int = 30, n_dti: int = 20) -> np.ndarray:
    """Fill gaps shorter than ``n_dti`` frames in tracks with more than ``n_min`` boxes.

    ``rows``: ``(K, 6) [frame, id, x, y, w, h]``; returns rows sorted by frame.
    """
    if len(rows) == 0:
        return rows
    out = []
    for track_id in np.unique(rows[:, 1]):
        tracklet = rows[rows[:, 1] == track_id]
        tracklet = tracklet[np.argsort(tracklet[:, 0], kind="stable")]
        out.append(tracklet)
        if len(tracklet) <= n_min:
            continue
        frames = tracklet[:, 0]
        filled = {}
        for i in range(1, len(tracklet)):
            left_frame, right_frame = frames[i - 1], frames[i]
            if 1 < right_frame - left_frame < n_dti:
                left_box, right_box = tracklet[i - 1, 2:6], tracklet[i, 2:6]
                for j in range(1, int(right_frame - left_frame - 1) + 1):
                    cur = j + left_frame
                    filled[cur] = (cur - left_frame) * (right_box - left_box) / (right_frame - left_frame) + left_box
        if filled:
            out.append(np.array([[f, track_id, *box] for f, box in filled.items()]))
    result = np.vstack(out)
    return result[np.argsort(result[:, 0], kind="stable")]
