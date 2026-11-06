"""Train a detector from a detector config.

Example:
    python scripts/train_detector.py configs/detectors/yolo11s_sweetfish.yaml
    python scripts/train_detector.py configs/detectors/yolo11s_sweetfish.yaml train.epochs=5

Weights land in weights/<model>/<dataset>/<run_id>/weights/best.pt.
"""
import argparse
import logging

from cmot.core.config import apply_overrides, load_yaml
from cmot.detection.pipeline import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="detector YAML (configs/detectors/...)")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = train_detector(apply_overrides(load_yaml(args.config), args.overrides))
    print(f"\nDetector run: {run_dir}")
    print(f"Next: python scripts/export_detections.py {run_dir} --splits <split> [<split> ...]")


if __name__ == "__main__":
    main()
