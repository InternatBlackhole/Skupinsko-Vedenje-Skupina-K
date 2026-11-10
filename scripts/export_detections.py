"""Run a trained detector over dataset splits and cache the detections for tracking.

Example:
    python scripts/export_detections.py weights/yolo11s/sweetfish/<run_id> --splits val_half
    python scripts/run_tracking.py configs/experiments/ocsort_sweetfish_yolo.yaml detections.det_id=<det_id>
"""
import argparse
import logging
from pathlib import Path

from cmot.detection.pipeline import export_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detector_dir", type=Path, help="weights/<model>/<dataset>/<run_id>")
    parser.add_argument("--splits", nargs="+", required=True, help="dataset splits to run on")
    parser.add_argument("--weights", default="best.pt", help="checkpoint inside <detector_dir>/weights/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_detections(args.detector_dir, args.splits, args.weights)


if __name__ == "__main__":
    main()
