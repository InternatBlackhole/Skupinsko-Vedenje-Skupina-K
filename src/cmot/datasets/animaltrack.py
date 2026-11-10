"""AnimalTrack (Zhang et al., 2022).

Raw layout (``<data_root>/AnimalTrack``)::

    Whole_AnimalTrack/videos_all/<name>.mp4
    Whole_AnimalTrack/gt_all/<name>_gt.txt        MOT format, comma separated
    train_test_splits/{videos,gt}_{train,test}.txt

Sequence names are the video stems, e.g. ``zebra_3``. Splits ``train`` / ``test``
default to the official split files unless listed explicitly in the config.
"""
import numpy as np

from cmot.datasets.base import Dataset, Sequence
from cmot.datasets.mot_txt import read_mot_txt


class AnimalTrack(Dataset):
    def sequences(self, split: str):
        if self.cfg.get("splits", {}).get(split) is None:
            split_file = self.root / "train_test_splits" / f"gt_{split}.txt"
            if not split_file.exists():
                raise KeyError(f"AnimalTrack: unknown split {split!r}")
            names = [line.strip().removesuffix("_gt.txt") for line in split_file.read_text().split() if line.strip()]
            return [self.load_sequence(n) for n in names]
        return super().sequences(split)

    def load_sequence(self, name: str) -> Sequence:
        whole = self.root / "Whole_AnimalTrack"
        gt = read_mot_txt(whole / "gt_all" / f"{name}_gt.txt", ignore_zero_conf=True)
        return Sequence(
            name=name,
            length=int(np.max(gt[:, 0])),
            gt=gt,
            meta={"video": str(whole / "videos_all" / f"{name}.mp4")},
        )
