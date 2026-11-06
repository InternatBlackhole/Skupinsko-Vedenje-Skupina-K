"""Detections cached on disk.

Location: ``outputs/detections/<dataset>/<split>/<det_id>/<seq>.txt`` with one
detection per line: ``frame,x1,y1,x2,y2,score`` (original image pixels; for
YOLOX the score is ``obj_conf * cls_conf`` as in the official OC-SORT).
"""
from pathlib import Path

import numpy as np

from cmot.core.paths import detections_dir
from cmot.datasets.base import Sequence
from cmot.detection.base import DetectionSource, group_by_frame


def write_detections(path: str | Path, per_frame: dict[int, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame in sorted(per_frame):
            for x1, y1, x2, y2, score in per_frame[frame]:
                f.write(f"{frame},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{score:.5f}\n")


class CachedDetections(DetectionSource):
    def __init__(self, dataset: str, split: str, det_id: str):
        self.det_id = det_id
        self.folder = detections_dir(dataset, split, det_id)

    @property
    def id(self) -> str:
        return self.det_id

    def load(self, seq: Sequence) -> dict[int, np.ndarray]:
        path = self.folder / f"{seq.name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"No cached detections for {seq.name}: {path}")
        if path.stat().st_size == 0:
            return {}
        return group_by_frame(np.loadtxt(path, delimiter=",", ndmin=2))
