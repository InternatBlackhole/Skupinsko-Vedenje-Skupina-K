"""Run an experiment config.

Example:
    python scripts/run_tracking.py configs/experiments/ocsort_sweetfish_oracle.yaml
    python scripts/run_tracking.py configs/experiments/ocsort_sweetfish_oracle.yaml \
        split=val_half tracker.overrides.inertia=0.1 detections.drop_rate=0.2
"""
import argparse
import logging

from cmot.core.config import apply_overrides, load_yaml, resolve_experiment
from cmot.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="experiment YAML (configs/experiments/...)")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides, applied before resolving")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = resolve_experiment(apply_overrides(load_yaml(args.config), args.overrides))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
