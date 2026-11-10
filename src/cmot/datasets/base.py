"""Common dataset representation.

Every dataset reader turns its raw files into ``Sequence`` objects with ground
truth in one format, so detections, tracking and evaluation never need to know
where the data came from. Frames are numbered from 1.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cmot.core.paths import data_root

GT_COLUMNS = ("frame", "id", "x", "y", "w", "h")  # box in tlwh pixels


@dataclass
class Sequence:
    name: str
    length: int                    # frames are 1..length
    gt: np.ndarray | None          # (K, 6) rows of GT_COLUMNS, or None if unannotated
    meta: dict = field(default_factory=dict)


def crop_frames(gt: np.ndarray, first: int, last: int) -> np.ndarray:
    """Keep frames ``first..last`` (inclusive) and renumber them from 1."""
    keep = (gt[:, 0] >= first) & (gt[:, 0] <= last)
    out = gt[keep].copy()
    out[:, 0] -= first - 1
    return out


class Dataset(ABC):
    """Reads one dataset; ``cfg`` is the dict from configs/datasets/<name>.yaml."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name: str = cfg["name"]
        self.root = Path(data_root()) / cfg["root"]

    def split_names(self) -> list[str]:
        return list(self.cfg["splits"])

    def sequences(self, split: str) -> list[Sequence]:
        if split not in self.cfg["splits"]:
            raise KeyError(f"{self.name}: unknown split {split!r}; available: {self.split_names()}")
        return [self._load_spec(spec) for spec in self.cfg["splits"][split]]

    def _load_spec(self, spec: str | dict) -> Sequence:
        """A split entry is a sequence name, or ``{seq: name, frames: [first, last]}``."""
        if isinstance(spec, str):
            return self.load_sequence(spec)
        seq = self.load_sequence(spec["seq"])
        if "frames" not in spec:
            return seq
        first, last = spec["frames"]
        gt = crop_frames(seq.gt, first, last) if seq.gt is not None else None
        meta = {**seq.meta, "frame_offset": first - 1}
        name = spec.get("name", f"{seq.name}_f{first}-{last}")
        return Sequence(name=name, length=last - first + 1, gt=gt, meta=meta)

    @abstractmethod
    def load_sequence(self, name: str) -> Sequence:
        """Load one full sequence by its name in this dataset."""
