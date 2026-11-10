"""SweetFish.

Raw layout (``<data_root>/SweetFish``)::

    Train/train.mp4  Train/train_gt_mot.txt
    Development/development.mp4
    Test/test.mp4

Only the train video is annotated (MOT format, whitespace separated, fixed
100x100 boxes). Sequence names: ``train``, ``development``, ``test``.

The GT file stores the box corner with x and y swapped: its "x" column is the
image row and its "y" column the image column (checked against fish positions
in the video: median error 68 px as swapped vs 587 px as given). ``swap_xy: true``
in the dataset config converts it to normal image coordinates on load.
"""
import numpy as np

from cmot.datasets.base import Dataset, Sequence
from cmot.datasets.mot_txt import read_mot_txt

_FILES = {
    "train": ("Train/train.mp4", "Train/train_gt_mot.txt"),
    "development": ("Development/development.mp4", None),
    "test": ("Test/test.mp4", None),
}


class SweetFish(Dataset):
    def load_sequence(self, name: str) -> Sequence:
        if name not in _FILES:
            raise KeyError(f"SweetFish has no sequence {name!r}; available: {sorted(_FILES)}")
        video, gt_file = _FILES[name]
        gt = read_mot_txt(self.root / gt_file) if gt_file else None
        if gt is not None and self.cfg.get("swap_xy", False):
            gt[:, [2, 3]] = gt[:, [3, 2]]
            gt[:, [4, 5]] = gt[:, [5, 4]]
        length = self.cfg.get("lengths", {}).get(name)
        if length is None:
            if gt is None:
                raise ValueError(f"SweetFish/{name}: no GT, set lengths.{name} in the dataset config")
            length = int(np.max(gt[:, 0]))
        return Sequence(name=name, length=int(length), gt=gt, meta={"video": str(self.root / video)})
