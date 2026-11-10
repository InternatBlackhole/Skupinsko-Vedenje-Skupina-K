"""OC-SORT parameters. Defaults are the paper / official repository defaults."""
from dataclasses import asdict, dataclass, field


@dataclass
class OCSortConfig:
    det_thresh: float = 0.6       # high-confidence detections (official --track_thresh)
    low_thresh: float = 0.1       # lower bound for the BYTE round's low-confidence detections
    max_age: int = 30             # frames a lost track is kept (official --track_buffer)
    min_hits: int = 3             # consecutive hits before a track is reported
    iou_threshold: float = 0.3
    delta_t: int = 3              # observation time gap used to estimate direction (OCM)
    inertia: float = 0.2          # weight of the OCM direction term
    asso_func: str = "iou"        # similarity for BYTE / OCR rounds: iou, giou, diou, ciou, ct_dist
    use_byte: bool = False
    use_oru: bool = True
    oru_trajectory: str = "linear"
    # "observation" reports the last associated detection (official default);
    # "state" reports the motion model's estimate instead.
    output_box: str = "observation"
    motion: dict = field(default_factory=lambda: {"name": "kf_cv_xysr"})

    @classmethod
    def from_dict(cls, cfg: dict) -> "OCSortConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(cfg) - known - {"name"}
        if unknown:
            raise KeyError(f"Unknown OC-SORT config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in cfg.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
