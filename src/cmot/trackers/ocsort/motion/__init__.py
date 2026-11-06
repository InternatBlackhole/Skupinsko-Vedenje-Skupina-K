"""OC-SORT motion models, selected by ``name`` in configs/trackers/ocsort/motion/*.yaml."""
from cmot.trackers.ocsort.motion.base import InteractionModel, MotionModel, NoInteraction
from cmot.trackers.ocsort.motion.kf_cv_xysr import KFConstantVelocityXYSR

MOTION_MODELS: dict[str, type[MotionModel]] = {
    "kf_cv_xysr": KFConstantVelocityXYSR,
}

INTERACTION_MODELS: dict[str, type[InteractionModel]] = {
    "none": NoInteraction,
}


def build_motion_factory(cfg: dict):
    """Return ``box -> MotionModel`` for the configured model and its parameters."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    cfg.pop("interaction", None)
    if name not in MOTION_MODELS:
        raise KeyError(f"Unknown OC-SORT motion model {name!r}; available: {sorted(MOTION_MODELS)}")
    cls = MOTION_MODELS[name]
    return lambda box: cls(box, **cfg)


def build_interaction(cfg: dict) -> InteractionModel:
    spec = dict(cfg.get("interaction") or {"name": "none"})
    name = spec.pop("name")
    if name not in INTERACTION_MODELS:
        raise KeyError(f"Unknown interaction model {name!r}; available: {sorted(INTERACTION_MODELS)}")
    return INTERACTION_MODELS[name](**spec)
