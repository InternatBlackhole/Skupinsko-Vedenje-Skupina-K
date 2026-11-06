"""Bounding-box format conversions.

Conventions used across the code base:
    xyxy : [x1, y1, x2, y2]   (top-left, bottom-right corners)
    tlwh : [x1, y1, w, h]     (MOTChallenge file format)
"""
import numpy as np


def tlwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    out = boxes.copy()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    out = boxes.copy()
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out
