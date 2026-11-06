"""HOTA / CLEAR / Identity metrics via TrackEval (MOTChallenge 2D box protocol).

Ground truth of any dataset is written in MOTChallenge format to
``outputs/prepared/<dataset>/<split>/<seq>/gt/gt.txt`` (every object as class 1,
"pedestrian", which is the only class TrackEval's MOT loader evaluates).
"""
from pathlib import Path

import numpy as np

from cmot.core.paths import prepared_dir
from cmot.datasets.base import Sequence


def _import_trackeval():
    # TrackEval still uses numpy aliases removed in numpy 1.24 (np.float, np.int, np.bool).
    for alias, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    import trackeval

    return trackeval


def prepare_gt(dataset: str, split: str, sequences: list[Sequence]) -> Path:
    """Write GT for TrackEval; derived from the raw data, so safe to regenerate."""
    root = prepared_dir(dataset) / split
    for seq in sequences:
        if seq.gt is None:
            raise ValueError(f"{dataset}/{seq.name} has no ground truth to evaluate against")
        gt_dir = root / seq.name / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        with open(gt_dir / "gt.txt", "w") as f:
            for frame, tid, x, y, w, h in seq.gt:
                f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
    return root


def evaluate_run(run_dir: Path, gt_root: Path, seq_lengths: dict[str, int], do_preproc: bool = False) -> dict:
    """Evaluate ``run_dir/tracks/*.txt``; writes TrackEval output to ``run_dir/metrics``.

    Returns ``{"COMBINED_SEQ": {...}, <seq>: {...}}`` with HOTA, DetA, AssA, MOTA,
    IDF1 (percent) and IDSW.
    """
    trackeval = _import_trackeval()
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(exist_ok=True)

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        PRINT_CONFIG=False,
        TIME_PROGRESS=False,
        PLOT_CURVES=False,
        PRINT_ONLY_COMBINED=True,
        LOG_ON_ERROR=str(run_dir / "metrics" / "error_log.txt"),
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        GT_FOLDER=str(gt_root),
        TRACKERS_FOLDER=str(run_dir.parent),
        OUTPUT_FOLDER=str(run_dir.parent),
        TRACKERS_TO_EVAL=[run_dir.name],
        TRACKER_SUB_FOLDER="tracks",
        OUTPUT_SUB_FOLDER="metrics",
        SEQ_INFO=dict(seq_lengths),
        SKIP_SPLIT_FOL=True,
        DO_PREPROC=do_preproc,
        PRINT_CONFIG=False,
    )
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    results, messages = evaluator.evaluate([dataset], metrics)
    status = messages["MotChallenge2DBox"][run_dir.name]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")

    out = {}
    for seq, res in results["MotChallenge2DBox"][run_dir.name].items():
        r = res["pedestrian"]
        out[seq] = {
            "HOTA": float(np.mean(r["HOTA"]["HOTA"]) * 100),
            "DetA": float(np.mean(r["HOTA"]["DetA"]) * 100),
            "AssA": float(np.mean(r["HOTA"]["AssA"]) * 100),
            "MOTA": float(r["CLEAR"]["MOTA"] * 100),
            "IDF1": float(r["Identity"]["IDF1"] * 100),
            "IDSW": int(r["CLEAR"]["IDSW"]),
        }
    return out
