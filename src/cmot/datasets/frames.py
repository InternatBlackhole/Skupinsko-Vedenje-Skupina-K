"""Reading sequence frames from a video file or an image folder.

Frame numbers are the sequence's own (1-based, after any ``frames: [first, last]``
cropping); ``meta["frame_offset"]`` maps them back to the raw video / image index.
"""
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from cmot.datasets.base import Sequence


def iter_frames(seq: Sequence, frames: Iterable[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame, BGR image)`` for the requested frames in increasing order."""
    wanted = sorted(set(frames)) if frames is not None else list(range(1, seq.length + 1))
    if "video" in seq.meta:
        yield from _iter_video(Path(seq.meta["video"]), wanted, seq.meta.get("frame_offset", 0))
    elif "img_dir" in seq.meta:
        yield from _iter_images(Path(seq.meta["img_dir"]), wanted, seq.meta.get("frame_offset", 0))
    else:
        raise ValueError(f"{seq.name}: no 'video' or 'img_dir' in sequence meta")


def frame_path(seq: Sequence, frame: int) -> Path:
    """Image file of ``frame`` for image-folder sequences (``%06d.jpg``, as in 3D-ZeF / MOT)."""
    return Path(seq.meta["img_dir"]) / f"{frame + seq.meta.get('frame_offset', 0):06d}.jpg"


def _iter_video(path: Path, wanted: list[int], offset: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    try:
        # Decode sequentially; seeking by frame index is not exact for every codec.
        targets = iter(wanted)
        target = next(targets, None)
        index = 0  # 1-based index of the last frame grabbed, in sequence numbering
        while target is not None:
            if not cap.grab():
                raise RuntimeError(f"{path}: video ended before frame {target + offset}")
            index += 1
            if index - offset != target:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{path}: failed to decode frame {target + offset}")
            yield target, image
            target = next(targets, None)
    finally:
        cap.release()


def _iter_images(img_dir: Path, wanted: list[int], offset: int):
    for frame in wanted:
        path = img_dir / f"{frame + offset:06d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {path}")
        yield frame, image
