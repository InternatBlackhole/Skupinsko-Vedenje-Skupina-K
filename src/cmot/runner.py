"""Runs one experiment: dataset split x detections x tracker (+ motion model) -> run dir + metrics."""
import json
import logging
import time
from pathlib import Path

import numpy as np

from cmot.core.boxes import xyxy_to_tlwh
from cmot.datasets import build_dataset
from cmot.datasets.mot_txt import write_tracks
from cmot.detection import EMPTY, build_detection_source
from cmot.evaluation.trackeval_runner import evaluate_run, prepare_gt
from cmot.postprocess.filters import filter_boxes
from cmot.postprocess.interpolation import interpolate_tracks
from cmot.trackers.base import Tracker
from cmot.trackers.registry import build_tracker
from cmot.utils.runs import append_to_index, create_run_dir

log = logging.getLogger(__name__)


def track_sequence(
    tracker: Tracker,
    dets_by_frame: dict[int, np.ndarray],
    length: int,
    skip_empty_frames: bool = False,
    min_box_area: float = 0.0,
    max_aspect_ratio: float | None = None,
) -> np.ndarray:
    """Run ``tracker`` over frames 1..length; returns ``(K, 6) [frame, id, x, y, w, h]``.

    ``skip_empty_frames`` reproduces the official evaluation scripts, which do not
    call the tracker on frames without any detection (tracks then do not age).
    """
    tracker.reset()
    rows = []
    for frame in range(1, length + 1):
        dets = dets_by_frame.get(frame, EMPTY)
        if len(dets) == 0 and skip_empty_frames:
            continue
        out = filter_boxes(tracker.update(dets), min_box_area, max_aspect_ratio)
        if len(out):
            tlwh = xyxy_to_tlwh(out[:, :4])
            rows.append(np.column_stack([np.full(len(out), frame), out[:, 4], tlwh]))
    return np.vstack(rows) if rows else np.empty((0, 6))


def run_experiment(cfg: dict) -> Path:
    """``cfg`` is a resolved experiment config (see ``cmot.core.config.resolve_experiment``)."""
    ds_cfg = cfg["dataset"]
    split = cfg["split"]
    dataset = build_dataset(ds_cfg)
    sequences = dataset.sequences(split)
    det_source = build_detection_source(cfg["detections"], dataset.name, split)
    tracker_cfg = cfg["tracker"]
    tracker = build_tracker(tracker_cfg)
    motion_name = tracker_cfg.get("motion", {}).get("name", "default")

    run_dir = create_run_dir(cfg, dataset.name, split, tracker_cfg["name"], motion_name)
    log.info("Run directory: %s", run_dir)

    runner_cfg = cfg.get("runner", {})
    out_filter = ds_cfg.get("output_filter", {})
    interp = runner_cfg.get("interpolation")
    timing = {}
    for seq in sequences:
        dets = det_source.load(seq)
        start = time.perf_counter()
        rows = track_sequence(
            tracker,
            dets,
            seq.length,
            skip_empty_frames=runner_cfg.get("skip_empty_frames", False),
            min_box_area=out_filter.get("min_box_area", 0.0),
            max_aspect_ratio=out_filter.get("max_aspect_ratio"),
        )
        elapsed = time.perf_counter() - start
        if interp:
            rows = interpolate_tracks(rows, **interp)
        write_tracks(run_dir / "tracks" / f"{seq.name}.txt", rows)
        timing[seq.name] = {"frames": seq.length, "seconds": round(elapsed, 3), "fps": round(seq.length / max(elapsed, 1e-9), 1)}
        log.info("%s: %d frames, %d boxes, %.1f FPS", seq.name, seq.length, len(rows), timing[seq.name]["fps"])
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))

    if cfg.get("evaluate", True) and all(s.gt is not None for s in sequences):
        gt_root = prepare_gt(dataset.name, split, sequences)
        metrics = evaluate_run(
            run_dir,
            gt_root,
            {s.name: s.length for s in sequences},
            do_preproc=ds_cfg.get("evaluation", {}).get("do_preproc", False),
        )
        (run_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))
        combined = metrics["COMBINED_SEQ"]
        log.info("HOTA %.2f | DetA %.2f | AssA %.2f | MOTA %.2f | IDF1 %.2f | IDSW %d",
                 combined["HOTA"], combined["DetA"], combined["AssA"], combined["MOTA"], combined["IDF1"], combined["IDSW"])
        append_to_index({
            "run_id": run_dir.name, "dataset": dataset.name, "split": split,
            "tracker": tracker_cfg["name"], "motion": motion_name, "detections": det_source.id,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in combined.items()},
            "run_dir": str(run_dir),
        })
    return run_dir
