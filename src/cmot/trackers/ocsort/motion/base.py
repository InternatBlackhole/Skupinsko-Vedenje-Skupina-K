"""Motion model interface for OC-SORT.

This is the focal point of the experiments: every motion model (the paper's
constant-velocity Kalman filter, constant acceleration, collective motion, ...)
implements ``MotionModel`` and is selected by name from the config. The OC-SORT
tracker only talks to this interface.

Two levels:

* ``MotionModel`` - one instance per track. Owns the state, predicts it forward
  and corrects it with observations. Works in its own measurement space ``z``
  (for the paper's model ``z = [cx, cy, area, aspect]``).
* ``InteractionModel`` - one instance per tracker. Looks at all tracks before the
  prediction step and returns a per-track control input (e.g. neighbour
  alignment for collective motion). The default returns no control, which is
  the original OC-SORT.
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class MotionModel(ABC):
    """Per-track motion model."""

    @abstractmethod
    def __init__(self, box_xyxy: np.ndarray, **params: Any) -> None:
        """Initialise the state from the first observed box."""

    # --- measurement space -------------------------------------------------
    @abstractmethod
    def box_to_measurement(self, box_xyxy: np.ndarray) -> np.ndarray:
        """Box ``[x1, y1, x2, y2]`` -> measurement column vector ``z``."""

    @abstractmethod
    def measurement_to_cxcywh(self, z: np.ndarray) -> tuple[float, float, float, float]:
        """Measurement -> ``(cx, cy, w, h)``; used to build ORU virtual trajectories."""

    @abstractmethod
    def cxcywh_to_measurement(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        """``(cx, cy, w, h)`` -> measurement column vector."""

    # --- filtering ---------------------------------------------------------
    @abstractmethod
    def predict(self, control: np.ndarray | None = None, guard: bool = True) -> None:
        """Advance the state by one frame.

        ``guard`` applies model-specific safety clamps (e.g. keeping the area
        positive). OC-SORT's ORU replay calls ``predict(guard=False)`` because
        the reference implementation steps the bare filter there.
        """

    @abstractmethod
    def update(self, z: np.ndarray | None) -> None:
        """Correct the state with measurement ``z``; ``None`` marks a missed frame."""

    @abstractmethod
    def box(self) -> np.ndarray:
        """Current state as a box ``[x1, y1, x2, y2]``, shape ``(4,)``."""

    def velocity(self) -> np.ndarray:
        """Centre velocity ``(vx, vy)`` in pixels per frame (for interaction models)."""
        raise NotImplementedError

    # --- ORU support -------------------------------------------------------
    @abstractmethod
    def snapshot(self) -> Any:
        """Return an opaque copy of everything needed to roll the state back."""

    @abstractmethod
    def restore(self, snap: Any) -> None:
        """Roll the state back to a previous ``snapshot()``."""


class InteractionModel(ABC):
    """Scene-level hook that gives each track a control input before prediction."""

    @abstractmethod
    def compute_controls(self, tracks: Sequence[Any]) -> list[np.ndarray | None]:
        """Return one control per track (``None`` = no control for that track)."""


class NoInteraction(InteractionModel):
    """Independent tracks, as in the original OC-SORT."""

    def compute_controls(self, tracks):
        return [None] * len(tracks)
