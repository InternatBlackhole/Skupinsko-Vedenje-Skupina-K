"""3D-ZeF (Pedersen et al., CVPR 2020), Hugging Face release ``vapaau/3D-ZeF``.

Raw layout (``<data_root>/3D-ZeF/data/ZebraFish-XX``)::

    gt/gt.txt    imgT/  imgF/  img1/ (both views side by side)
    cam{T,F}_intrinsic.json  cam{T,F}_references.json

Each GT row holds a 3D head position plus one 2D box per camera view::

    frame, id, 3d_x, 3d_y, 3d_z,
    camT_x, camT_y, camT_left, camT_top, camT_width, camT_height, camT_occlusion,
    camF_x, camF_y, camF_left, camF_top, camF_width, camF_height, camF_occlusion

We track each view as its own 2D sequence: ``ZebraFish-01-T`` (top) and
``ZebraFish-01-F`` (front).
"""
import numpy as np

from cmot.datasets.base import Dataset, Sequence

_BOX_COLUMNS = {"T": slice(7, 11), "F": slice(14, 18)}  # left, top, width, height


class ZeF3D(Dataset):
    def load_sequence(self, name: str) -> Sequence:
        base, view = name.rsplit("-", 1)
        if view not in _BOX_COLUMNS:
            raise KeyError(f"3D-ZeF sequence names end in -T or -F, got {name!r}")
        seq_dir = self.root / "data" / base
        raw = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
        gt = np.column_stack([raw[:, 0], raw[:, 1], raw[:, _BOX_COLUMNS[view]]])
        gt = gt[(gt[:, 4] > 0) & (gt[:, 5] > 0)]
        gt = gt[np.lexsort((gt[:, 1], gt[:, 0]))]

        img_dir = seq_dir / f"img{view}"
        n_images = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
        length = max(int(gt[:, 0].max()), n_images)
        return Sequence(name=name, length=length, gt=gt, meta={"img_dir": str(img_dir), "view": view})
