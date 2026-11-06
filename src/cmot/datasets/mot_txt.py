"""Readers / writers for MOTChallenge-style text files."""
from pathlib import Path

import numpy as np


def read_mot_txt(path: str | Path, ignore_zero_conf: bool = False) -> np.ndarray:
    """Read ``frame, id, x, y, w, h[, conf, ...]`` rows (comma or whitespace separated).

    Returns ``(K, 6)`` float array ``frame, id, x, y, w, h``. Boxes without a
    positive width and height are dropped.
    """
    text = Path(path).read_text().replace(",", " ")
    data = np.loadtxt(text.splitlines(), dtype=np.float64, ndmin=2)
    if data.size == 0:
        return np.empty((0, 6))
    if ignore_zero_conf and data.shape[1] > 6:
        data = data[data[:, 6] != 0]
    data = data[:, :6]
    data = data[(data[:, 4] > 0) & (data[:, 5] > 0)]
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def write_tracks(path: str | Path, rows: np.ndarray) -> None:
    """Write tracker output ``frame, id, x, y, w, h`` like the official OC-SORT
    (``write_results_no_score``: boxes rounded to 0.1 px, trailing ``-1`` fields)."""
    with open(path, "w") as f:
        for frame, tid, x, y, w, h in rows:
            f.write(
                f"{int(frame)},{int(tid)},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},-1,-1,-1,-1\n"
            )


def read_tracks(path: str | Path) -> np.ndarray:
    """Read tracker output written by ``write_tracks`` -> ``(K, 6)``."""
    if Path(path).stat().st_size == 0:
        return np.empty((0, 6))
    return np.loadtxt(path, delimiter=",", ndmin=2)[:, :6]
