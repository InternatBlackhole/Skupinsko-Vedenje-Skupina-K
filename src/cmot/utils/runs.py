"""Run directories that never overwrite each other.

Layout::

    outputs/runs/<dataset>/<split>/<tracker>/<motion>/<run_id>/
        config.resolved.yaml   full config used for the run
        git.txt                commit, dirty flag and diff at run time
        tracks/<seq>.txt       tracker output (MOTChallenge format)
        metrics/               TrackEval output

``run_id = <timestamp>_<config hash>_<git sha>``. The directory is created with
``exist_ok=False``, so a collision raises instead of overwriting results.
"""
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from cmot.core.paths import REPO_ROOT, outputs_root, runs_dir


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info() -> dict:
    status = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "--short", "HEAD") or "nogit",
        "dirty": bool(status),
        "status": status,
        "diff": _git("diff", "HEAD"),
    }


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def new_run_id(cfg: dict) -> str:
    """``<timestamp>_<config hash>_<git sha>``; shared by tracking and detector runs."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{config_hash(cfg)}_{git_info()['sha']}"


def write_provenance(run_dir: Path, cfg: dict, config_name: str = "config.resolved.yaml") -> None:
    """Save the resolved config and the git state next to a run's outputs."""
    info = git_info()
    with open(run_dir / config_name, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "git.txt", "w") as f:
        f.write(f"sha: {info['sha']}\ndirty: {info['dirty']}\n\n# git status\n{info['status']}\n\n# git diff HEAD\n{info['diff']}\n")


def create_run_dir(cfg: dict, dataset: str, split: str, tracker: str, motion: str) -> Path:
    run_dir = runs_dir() / dataset / split / tracker / motion / new_run_id(cfg)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tracks").mkdir()
    write_provenance(run_dir, cfg)
    return run_dir


INDEX_FIELDS = [
    "run_id", "dataset", "split", "tracker", "motion", "detections",
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "run_dir",
]


def append_to_index(row: dict) -> None:
    """Append one line per run to outputs/index.csv (never rewritten)."""
    index = outputs_root() / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    new = not index.exists()
    with open(index, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
