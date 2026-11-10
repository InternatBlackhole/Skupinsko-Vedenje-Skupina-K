"""Output filters applied by the official evaluation scripts, outside the tracker.

The official MOT17 evaluator keeps a box only if ``w * h > min_box_area`` (100)
and ``w / h <= 1.6``; DanceTrack uses only the area filter. These are
pedestrian-specific, so they are set per dataset (off for fish).
"""
import numpy as np


def filter_boxes(tracks: np.ndarray, min_box_area: float = 0.0, max_aspect_ratio: float | None = None) -> np.ndarray:
    """``tracks``: ``(M, 5) [x1, y1, x2, y2, id]``."""
    if len(tracks) == 0:
        return tracks
    w = tracks[:, 2] - tracks[:, 0]
    h = tracks[:, 3] - tracks[:, 1]
    keep = w * h > min_box_area
    if max_aspect_ratio is not None:
        keep &= ~(w / h > max_aspect_ratio)
    return tracks[keep]
