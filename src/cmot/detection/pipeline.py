"""Detector training and detection export.

Train:   configs/detectors/<name>.yaml -> weights/<model>/<dataset>/<run_id>/
             detector.yaml      resolved detector config (+ dataset config)
             git.txt            code state at training time
             weights/best.pt    Ultralytics checkpoint (plus its training logs/plots)

Export:  weights run + split  -> outputs/detections/<dataset>/<split>/<det_id>/
             <seq>.txt          frame,x1,y1,x2,y2,score
             meta.yaml          weights, inference settings, timing

``det_id = <model>_<run_id>``, so detections from different weights never mix.
Both steps refuse to overwrite an existing folder.
"""
import logging
import shutil
import time
from pathlib import Path

import yaml

from cmot.core.config import load_yaml, resolve_component
from cmot.core.paths import detections_dir, weights_root
from cmot.datasets import build_dataset
from cmot.detection.cached import write_detections
from cmot.detection.ultralytics_detector import UltralyticsDetector
from cmot.detection.yolo_dataset import export_yolo_dataset
from cmot.utils.runs import new_run_id, write_provenance

log = logging.getLogger(__name__)


def _model_stem(cfg: dict) -> str:
    return Path(cfg["model"]).stem


def train_detector(cfg: dict) -> Path:
    """Train an Ultralytics detector from a detector config; returns the weights run dir."""
    if cfg.get("framework", "ultralytics") != "ultralytics":
        raise NotImplementedError(f"Unsupported detector framework {cfg['framework']!r}")
    from ultralytics import YOLO

    cfg = dict(cfg, dataset=resolve_component("datasets", cfg["dataset"]))
    dataset = build_dataset(cfg["dataset"])
    data = cfg["data"]
    data_yaml = export_yolo_dataset(
        dataset, data["train_split"], data["val_split"], data.get("frame_stride", 1), data.get("class_name", "object")
    )

    run_id = new_run_id(cfg)
    project = weights_root() / _model_stem(cfg) / dataset.name
    run_dir = project / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_provenance(run_dir, cfg, config_name="detector.yaml")

    # Pretrained checkpoints are downloaded into weights/pretrained, never into the repo root.
    pretrained = weights_root() / "pretrained" / Path(cfg["model"]).name
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    model.train(
        data=str(data_yaml),
        project=str(project),
        name=run_id,
        exist_ok=True,  # the folder was created above (and only holds our provenance files)
        single_cls=True,
        **cfg.get("train", {}),
    )
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished without {best}")
    log.info("Trained detector: %s", best)
    return run_dir


def export_detections(detector_dir: Path, splits: list[str], weights_file: str = "best.pt") -> list[Path]:
    """Run a trained detector over dataset splits and cache the detections."""
    detector_dir = Path(detector_dir)
    cfg = load_yaml(detector_dir / "detector.yaml")
    dataset = build_dataset(cfg["dataset"])
    det_id = f"{_model_stem(cfg)}_{detector_dir.name}"
    infer = dict(cfg.get("inference", {}))
    detector = UltralyticsDetector(str(detector_dir / "weights" / weights_file), **infer)

    outputs = []
    for split in splits:
        out = detections_dir(dataset.name, split, det_id)
        if out.exists():
            raise FileExistsError(f"Detections already cached, refusing to overwrite: {out}")
        partial = out.with_name(out.name + ".partial")  # renamed only once the split is complete
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        timing = {}
        for seq in dataset.sequences(split):
            start = time.perf_counter()
            dets = detector.detect_sequence(seq, log=log)
            elapsed = time.perf_counter() - start
            write_detections(partial / f"{seq.name}.txt", dets)
            timing[seq.name] = {"frames": seq.length, "fps": round(seq.length / max(elapsed, 1e-9), 1)}
            log.info("%s/%s: %d frames with detections, %.1f FPS", split, seq.name, len(dets), timing[seq.name]["fps"])
        with open(partial / "meta.yaml", "w") as f:
            yaml.safe_dump(
                {"det_id": det_id, "detector_dir": str(detector_dir), "weights": weights_file,
                 "inference": infer, "timing": timing},
                f, sort_keys=False,
            )
        partial.rename(out)
        log.info("Detections cached: %s  (use detections.det_id=%s)", out, det_id)
        outputs.append(out)
    return outputs
