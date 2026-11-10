"""YAML config loading.

An experiment config references component configs by name, e.g.::

    dataset: sweetfish                 -> configs/datasets/sweetfish.yaml
    tracker:
      name: ocsort
      config: default                  -> configs/trackers/ocsort/default.yaml
      motion: kf_cv_xysr               -> configs/trackers/ocsort/motion/kf_cv_xysr.yaml

``resolve_experiment`` expands these references into one self-contained dict,
which is what gets saved next to every run.
"""
import copy
from pathlib import Path
from typing import Any

import yaml

from cmot.core.paths import configs_root


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` style overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value: Any = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def resolve_component(subdir: str, name_or_dict: str | dict) -> dict:
    if isinstance(name_or_dict, dict):
        return copy.deepcopy(name_or_dict)
    return load_yaml(configs_root() / subdir / f"{name_or_dict}.yaml")


def resolve_experiment(cfg: dict) -> dict:
    """Expand named component references into full config dicts."""
    cfg = copy.deepcopy(cfg)

    cfg["dataset"] = resolve_component("datasets", cfg["dataset"])

    trk = cfg["tracker"]
    name = trk["name"]
    tracker_cfg = resolve_component(f"trackers/{name}", trk.get("config", "default"))
    # The tracker's default config names its motion model; the experiment may pick another.
    motion_ref = trk.get("motion", tracker_cfg.get("motion"))
    if motion_ref is not None:
        tracker_cfg["motion"] = resolve_component(f"trackers/{name}/motion", motion_ref)
    tracker_cfg = deep_merge(tracker_cfg, trk.get("overrides", {}))
    cfg["tracker"] = {"name": name, **tracker_cfg}
    return cfg
