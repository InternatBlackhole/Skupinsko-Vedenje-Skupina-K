"""Central place for filesystem locations.

Datasets live outside the repository (default ``~/Desktop/Data/<dataset>``) and
are treated as read-only. Everything generated goes to ``<repo>/outputs``.
Both roots can be overridden with environment variables.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return Path(os.environ.get("CMOT_DATA_ROOT", Path.home() / "Desktop" / "Data")).expanduser()


def outputs_root() -> Path:
    return Path(os.environ.get("CMOT_OUTPUTS_ROOT", REPO_ROOT / "outputs")).expanduser()


def configs_root() -> Path:
    return REPO_ROOT / "configs"


def prepared_dir(dataset: str) -> Path:
    """Derived files for a dataset (e.g. ground truth converted for TrackEval)."""
    return outputs_root() / "prepared" / dataset


def detections_dir(dataset: str, split: str, det_id: str) -> Path:
    """Cached detections for one (dataset, split, detector weights) combination."""
    return outputs_root() / "detections" / dataset / split / det_id


def runs_dir() -> Path:
    return outputs_root() / "runs"


def weights_root() -> Path:
    """Detector weights: ``weights/<model>/<dataset>/<train_run_id>/`` (+ ``weights/pretrained``)."""
    return Path(os.environ.get("CMOT_WEIGHTS_ROOT", REPO_ROOT / "weights")).expanduser()
