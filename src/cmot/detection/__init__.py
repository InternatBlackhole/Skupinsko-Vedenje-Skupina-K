"""Detection sources, selected by ``detections.source`` in an experiment config."""
from cmot.detection.base import EMPTY, DetectionSource
from cmot.detection.cached import CachedDetections, write_detections
from cmot.detection.gt_oracle import GTOracle


def build_detection_source(cfg: dict, dataset: str, split: str) -> DetectionSource:
    cfg = dict(cfg)
    source = cfg.pop("source")
    if source == "gt_oracle":
        return GTOracle(**cfg)
    if source == "cached":
        if not cfg.get("det_id"):
            raise ValueError("cached detections need an id: pass detections.det_id=<id printed by export_detections.py>")
        return CachedDetections(dataset, split, cfg["det_id"])
    raise KeyError(f"Unknown detection source {source!r}; available: gt_oracle, cached")


__all__ = ["EMPTY", "DetectionSource", "build_detection_source", "write_detections"]
