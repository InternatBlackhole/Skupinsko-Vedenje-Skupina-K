"""Dataset readers, selected by ``reader`` in configs/datasets/<name>.yaml."""
from cmot.datasets.animaltrack import AnimalTrack
from cmot.datasets.base import Dataset, Sequence
from cmot.datasets.sweetfish import SweetFish
from cmot.datasets.zef3d import ZeF3D

READERS: dict[str, type[Dataset]] = {
    "sweetfish": SweetFish,
    "zef3d": ZeF3D,
    "animaltrack": AnimalTrack,
}


def build_dataset(cfg: dict) -> Dataset:
    reader = cfg.get("reader", cfg["name"])
    if reader not in READERS:
        raise KeyError(f"Unknown dataset reader {reader!r}; available: {sorted(READERS)}")
    return READERS[reader](cfg)


__all__ = ["Dataset", "Sequence", "build_dataset"]
