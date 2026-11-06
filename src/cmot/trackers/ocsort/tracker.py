"""OC-SORT: Observation-Centric SORT (Cao et al., CVPR 2023, arXiv:2203.14360).

Per frame:
    1. split detections by score (high > det_thresh; low in (low_thresh, det_thresh))
    2. interaction model gives per-track controls; every track predicts with its motion model
    3. stage 1: high dets vs predicted boxes, IoU + OCM direction term
    4. (optional) BYTE: low dets vs remaining predicted boxes, similarity only
    5. stage 3, OCR: remaining high dets vs the *last observations* of remaining tracks
    6. unmatched tracks get a missed update (ORU freezes their state);
       unmatched high dets start new tracks
    7. report confirmed tracks, delete tracks lost for more than max_age frames

This is a line-by-line port of the official ``OCSort.update`` (noahcao/OC_SORT),
restructured so that the motion model and the ORU trajectory are pluggable.
With the default config its outputs match the official code exactly.
"""
import numpy as np

from cmot.trackers.base import Tracker
from cmot.trackers.ocsort.association import SIMILARITY_FUNCS, associate_iou, associate_ocm
from cmot.trackers.ocsort.config import OCSortConfig
from cmot.trackers.ocsort.motion import build_interaction, build_motion_factory
from cmot.trackers.ocsort.observations import k_previous_obs
from cmot.trackers.ocsort.track import OCSortTrack


class OCSort(Tracker):
    def __init__(self, cfg: OCSortConfig | None = None):
        self.cfg = cfg or OCSortConfig()
        if self.cfg.output_box not in ("observation", "state"):
            raise ValueError(f"output_box must be 'observation' or 'state', got {self.cfg.output_box!r}")
        self.similarity = SIMILARITY_FUNCS[self.cfg.asso_func]
        self.motion_factory = build_motion_factory(self.cfg.motion)
        self.interaction = build_interaction(self.cfg.motion)
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict) -> "OCSort":
        return cls(OCSortConfig.from_dict(cfg))

    def reset(self) -> None:
        self.tracks: list[OCSortTrack] = []
        self.frame_count = 0
        self._next_id = 0

    def _new_track(self, det: np.ndarray) -> OCSortTrack:
        track = OCSortTrack(
            det,
            self._next_id,
            self.motion_factory,
            delta_t=self.cfg.delta_t,
            use_oru=self.cfg.use_oru,
            oru_trajectory=self.cfg.oru_trajectory,
        )
        self._next_id += 1
        return track

    def update(self, dets: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        self.frame_count += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)

        scores = dets[:, 4]
        dets_second = dets[np.logical_and(scores > cfg.low_thresh, scores < cfg.det_thresh)]
        dets = dets[scores > cfg.det_thresh]

        # --- predict ----------------------------------------------------------
        controls = self.interaction.compute_controls(self.tracks)
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        for t, track in enumerate(self.tracks):
            pos = track.predict(controls[t])
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        zero_vel = np.array((0, 0))
        velocities = np.array([t.velocity if t.velocity is not None else zero_vel for t in self.tracks])
        last_boxes = np.array([t.last_observation for t in self.tracks])
        k_observations = np.array(
            [k_previous_obs(t.observations, t.age, cfg.delta_t) for t in self.tracks]
        )

        # --- stage 1: IoU + OCM ----------------------------------------------
        matched, unmatched_dets, unmatched_trks = associate_ocm(
            dets, trks, cfg.iou_threshold, velocities, k_observations, cfg.inertia
        )
        for d, t in matched:
            self.tracks[t].update(dets[d, :])

        # --- stage 2 (optional): BYTE on low-score detections ----------------
        if cfg.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            pairs = associate_iou(dets_second, trks[unmatched_trks], cfg.iou_threshold, self.similarity)
            if pairs is not None:
                matched_trks = []
                for d, t in pairs:
                    trk_ind = unmatched_trks[t]
                    self.tracks[trk_ind].update(dets_second[d, :])
                    matched_trks.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(matched_trks, dtype=int))

        # --- stage 3: OCR, match against last observations -------------------
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            pairs = associate_iou(
                dets[unmatched_dets], last_boxes[unmatched_trks], cfg.iou_threshold, self.similarity
            )
            if pairs is not None:
                matched_dets, matched_trks = [], []
                for d, t in pairs:
                    det_ind, trk_ind = unmatched_dets[d], unmatched_trks[t]
                    self.tracks[trk_ind].update(dets[det_ind, :])
                    matched_dets.append(det_ind)
                    matched_trks.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(matched_dets, dtype=int))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(matched_trks, dtype=int))

        for t in unmatched_trks:
            self.tracks[t].update(None)

        for d in unmatched_dets:
            self.tracks.append(self._new_track(dets[d, :]))

        # --- output and track deletion ---------------------------------------
        ret = []
        i = len(self.tracks)
        for track in reversed(self.tracks):
            if cfg.output_box == "state" or track.last_observation.sum() < 0:
                box = track.state_box()
            else:
                box = track.last_observation[:4]
            if track.time_since_update < 1 and (
                track.hit_streak >= cfg.min_hits or self.frame_count <= cfg.min_hits
            ):
                ret.append(np.concatenate((box, [track.id + 1])).reshape(1, -1))  # MOT ids start at 1
            i -= 1
            if track.time_since_update > cfg.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
