"""Detections derived from ground truth, optionally corrupted.

Useful before a detector exists, and for isolating the effect of the motion
model from detector errors. Noise is reproducible: the random stream depends
only on ``seed`` and the sequence name.
"""
import zlib

import numpy as np

from cmot.core.boxes import tlwh_to_xyxy
from cmot.datasets.base import Sequence
from cmot.detection.base import DetectionSource, group_by_frame


class GTOracle(DetectionSource):
    def __init__(
        self,
        position_noise: float = 0.0,  # std of centre shift, as a fraction of box w / h
        size_noise: float = 0.0,      # std of relative change of w and h
        drop_rate: float = 0.0,       # probability a GT box is missed
        score: float = 1.0,           # confidence given to every detection
        seed: int = 0,
    ):
        self.position_noise = position_noise
        self.size_noise = size_noise
        self.drop_rate = drop_rate
        self.score = score
        self.seed = seed

    @property
    def id(self) -> str:
        if not (self.position_noise or self.size_noise or self.drop_rate):
            return "gt_oracle"
        return f"gt_oracle_p{self.position_noise}_s{self.size_noise}_d{self.drop_rate}_seed{self.seed}"

    def load(self, seq: Sequence) -> dict[int, np.ndarray]:
        if seq.gt is None:
            raise ValueError(f"{seq.name} has no ground truth; gt_oracle detections are impossible")
        rng = np.random.default_rng([self.seed, zlib.crc32(seq.name.encode())])
        gt = seq.gt
        x, y, w, h = gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5]
        cx = x + w / 2 + rng.normal(0.0, self.position_noise, len(gt)) * w
        cy = y + h / 2 + rng.normal(0.0, self.position_noise, len(gt)) * h
        w = w * np.maximum(1.0 + rng.normal(0.0, self.size_noise, len(gt)), 0.1)
        h = h * np.maximum(1.0 + rng.normal(0.0, self.size_noise, len(gt)), 0.1)
        keep = rng.random(len(gt)) >= self.drop_rate

        boxes = tlwh_to_xyxy(np.column_stack([cx - w / 2, cy - h / 2, w, h]))
        rows = np.column_stack([gt[:, 0], boxes, np.full(len(gt), self.score)])[keep]
        return group_by_frame(rows)
