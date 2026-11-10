"""Maps tracker names (``tracker.name`` in configs) to implementations."""
from cmot.trackers.base import Tracker
from cmot.trackers.ocsort import OCSort

TRACKERS: dict[str, type[Tracker]] = {
    "ocsort": OCSort,
}


def build_tracker(cfg: dict) -> Tracker:
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in TRACKERS:
        raise KeyError(f"Unknown tracker {name!r}; available: {sorted(TRACKERS)}")
    return TRACKERS[name].from_config(cfg)
