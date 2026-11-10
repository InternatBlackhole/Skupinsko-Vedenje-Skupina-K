"""Export dataset splits to the YOLO training format used by Ultralytics.

Output (derived data, reused if already complete)::

    outputs/prepared/<dataset>/yolo/<train>-<val>_stride<k>/
        images/{train,val}/<seq>_<frame>.jpg   video frames written as JPEG,
                                                image-folder frames symlinked
        labels/{train,val}/<seq>_<frame>.txt   "0 cx cy w h" (normalised)
        data.yaml                               written last = export complete

Every ``stride``-th frame is used; consecutive frames are near-duplicates.
Frames without objects are kept as background images.
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from cmot.core.paths import prepared_dir
from cmot.datasets.base import Dataset, Sequence
from cmot.datasets.frames import frame_path, iter_frames

log = logging.getLogger(__name__)


def yolo_labels(gt_rows: np.ndarray, img_w: int, img_h: int) -> list[str]:
    """GT rows ``frame, id, x, y, w, h`` -> YOLO label lines, boxes clipped to the image."""
    lines = []
    for _, _, x, y, w, h in gt_rows:
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, img_w), min(y + h, img_h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}")
    return lines


def _export_sequence(seq: Sequence, part: str, out: Path, stride: int) -> int:
    images_dir, labels_dir = out / "images" / part, out / "labels" / part
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(1, seq.length + 1, stride))
    by_frame = {int(f): seq.gt[seq.gt[:, 0] == f] for f in np.unique(seq.gt[:, 0])}
    empty = np.empty((0, 6))

    if "img_dir" in seq.meta:  # symlink originals instead of copying
        img_h, img_w = cv2.imread(str(frame_path(seq, 1))).shape[:2]
        for f in frames:
            stem = f"{seq.name}_{f:06d}"
            link = images_dir / f"{stem}.jpg"
            if not link.exists():
                os.symlink(frame_path(seq, f), link)
            (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_labels(by_frame.get(f, empty), img_w, img_h)))
        return len(frames)

    for f, image in iter_frames(seq, frames):
        stem = f"{seq.name}_{f:06d}"
        img_h, img_w = image.shape[:2]
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_labels(by_frame.get(f, empty), img_w, img_h)))
    return len(frames)


def export_yolo_dataset(dataset: Dataset, train_split: str, val_split: str, stride: int, class_name: str) -> Path:
    """Write (or reuse) the YOLO dataset for ``train_split`` / ``val_split``; returns data.yaml."""
    out = prepared_dir(dataset.name) / "yolo" / f"{train_split}-{val_split}_stride{stride}"
    data_yaml = out / "data.yaml"
    if data_yaml.exists():
        log.info("Reusing YOLO dataset %s", out)
        return data_yaml

    for part, split in (("train", train_split), ("val", val_split)):
        for seq in dataset.sequences(split):
            if seq.gt is None:
                raise ValueError(f"{dataset.name}/{seq.name} has no GT; cannot train a detector on it")
            n = _export_sequence(seq, part, out, stride)
            log.info("%s/%s -> %s: %d images", dataset.name, seq.name, part, n)

    with open(data_yaml, "w") as f:
        yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val", "names": {0: class_name}}, f)
    return data_yaml
