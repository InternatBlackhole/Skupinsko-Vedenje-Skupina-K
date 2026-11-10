"""Ultralytics YOLO inference over a whole sequence.

Produces detections in the tracker's format ``[x1, y1, x2, y2, score]`` in
original image pixels (Ultralytics rescales internally). YOLO has no separate
objectness, so the score is the class confidence.
"""
import numpy as np

from cmot.datasets.base import Sequence
from cmot.datasets.frames import iter_frames


class UltralyticsDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.05,      # keep low-score boxes for the tracker's BYTE / threshold logic
        iou: float = 0.7,        # NMS IoU
        quantize: int | str | None = 16,  # FP16 inference (Ultralytics >= 8.4; replaces "half")
        batch: int = 8,
        max_det: int = 300,
        device: str | int = 0,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.kwargs = dict(imgsz=imgsz, conf=conf, iou=iou, quantize=quantize, max_det=max_det, device=device, verbose=False)
        self.batch = batch

    def _predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for result in self.model.predict(images, **self.kwargs):
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
            score = boxes.conf.cpu().numpy().astype(np.float64)
            out.append(np.column_stack([xyxy, score]) if len(score) else np.empty((0, 5)))
        return out

    def detect_sequence(self, seq: Sequence, progress_every: int = 1000, log=None) -> dict[int, np.ndarray]:
        dets: dict[int, np.ndarray] = {}
        frames, images = [], []
        for frame, image in iter_frames(seq):
            frames.append(frame)
            images.append(image)
            if len(images) == self.batch:
                dets.update(zip(frames, self._predict(images)))
                frames, images = [], []
                if log and frame % progress_every < self.batch:
                    log.info("%s: frame %d / %d", seq.name, frame, seq.length)
        if images:
            dets.update(zip(frames, self._predict(images)))
        return {f: d for f, d in dets.items() if len(d)}
