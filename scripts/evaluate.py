"""Re-evaluate an existing run directory (e.g. after changing metric settings).

Example:
    python scripts/evaluate.py outputs/runs/sweetfish/val_half/ocsort/kf_cv_xysr/<run_id>
"""
import argparse
import json
from pathlib import Path

from cmot.core.config import load_yaml
from cmot.datasets import build_dataset
from cmot.evaluation.trackeval_runner import evaluate_run, prepare_gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    cfg = load_yaml(args.run_dir / "config.resolved.yaml")
    dataset = build_dataset(cfg["dataset"])
    sequences = dataset.sequences(cfg["split"])
    gt_root = prepare_gt(dataset.name, cfg["split"], sequences)
    metrics = evaluate_run(
        args.run_dir,
        gt_root,
        {s.name: s.length for s in sequences},
        do_preproc=cfg["dataset"].get("evaluation", {}).get("do_preproc", False),
    )
    (args.run_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["COMBINED_SEQ"], indent=2))


if __name__ == "__main__":
    main()
