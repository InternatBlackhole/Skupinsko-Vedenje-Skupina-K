"""Association for OC-SORT.

* similarity functions (IoU and variants), detections x tracks
* ``linear_assignment`` (Hungarian via ``lap.lapjv``, SciPy fallback)
* ``associate_ocm``  - stage 1: IoU + observation-centric momentum (OCM)
* ``associate_iou``  - plain IoU matching used by BYTE and by OCR (stage 3)

All functions follow the official implementation exactly, including the
shortcut that skips the Hungarian step when IoU alone is unambiguous.
"""
import numpy as np

from cmot.trackers.ocsort.observations import speed_direction_batch

# --- similarity --------------------------------------------------------------


def _intersection_union(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return bboxes1, bboxes2, wh, union


def iou_batch(bboxes1, bboxes2):
    _, _, wh, union = _intersection_union(bboxes1, bboxes2)
    return wh / union


def giou_batch(bboxes1, bboxes2):
    b1, b2, wh, union = _intersection_union(bboxes1, bboxes2)
    iou = wh / union
    wc = np.maximum(b1[..., 2], b2[..., 2]) - np.minimum(b1[..., 0], b2[..., 0])
    hc = np.maximum(b1[..., 3], b2[..., 3]) - np.minimum(b1[..., 1], b2[..., 1])
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    return (giou + 1.0) / 2.0  # (-1, 1) -> (0, 1)


def _centre_and_outer_diag(b1, b2):
    cx1 = (b1[..., 0] + b1[..., 2]) / 2.0
    cy1 = (b1[..., 1] + b1[..., 3]) / 2.0
    cx2 = (b2[..., 0] + b2[..., 2]) / 2.0
    cy2 = (b2[..., 1] + b2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    xxc1 = np.minimum(b1[..., 0], b2[..., 0])
    yyc1 = np.minimum(b1[..., 1], b2[..., 1])
    xxc2 = np.maximum(b1[..., 2], b2[..., 2])
    yyc2 = np.maximum(b1[..., 3], b2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    return inner_diag, outer_diag


def diou_batch(bboxes1, bboxes2):
    b1, b2, wh, union = _intersection_union(bboxes1, bboxes2)
    iou = wh / union
    inner_diag, outer_diag = _centre_and_outer_diag(b1, b2)
    diou = iou - inner_diag / outer_diag
    return (diou + 1) / 2.0


def ciou_batch(bboxes1, bboxes2):
    b1, b2, wh, union = _intersection_union(bboxes1, bboxes2)
    iou = wh / union
    inner_diag, outer_diag = _centre_and_outer_diag(b1, b2)
    w1 = b1[..., 2] - b1[..., 0]
    h1 = b1[..., 3] - b1[..., 1] + 1.0  # one-pixel shift avoids division by zero
    w2 = b2[..., 2] - b2[..., 0]
    h2 = b2[..., 3] - b2[..., 1] + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    return (ciou + 1) / 2.0


def ct_dist(bboxes1, bboxes2):
    """Centre distance rescaled to (0, 1). Coarse; the authors do not recommend it alone."""
    b2 = np.expand_dims(bboxes2, 0)
    b1 = np.expand_dims(bboxes1, 1)
    cx1 = (b1[..., 0] + b1[..., 2]) / 2.0
    cy1 = (b1[..., 1] + b1[..., 3]) / 2.0
    cx2 = (b2[..., 0] + b2[..., 2]) / 2.0
    cy2 = (b2[..., 1] + b2[..., 3]) / 2.0
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    dist = dist / dist.max()
    return dist.max() - dist


SIMILARITY_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "diou": diou_batch,
    "ciou": ciou_batch,
    "ct_dist": ct_dist,
}

# --- assignment --------------------------------------------------------------


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment; returns ``(K, 2)`` array of ``[row, col]`` pairs."""
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def _filter_matches(matched_indices, iou_matrix, iou_threshold, num_dets, num_trks):
    """Split assignments into matches / unmatched, rejecting pairs below the IoU threshold."""
    if matched_indices.shape[0] > 0:
        unmatched_dets = np.setdiff1d(np.arange(num_dets), matched_indices[:, 0])
        unmatched_trks = np.setdiff1d(np.arange(num_trks), matched_indices[:, 1])
        iou_vals = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]
        low = iou_vals < iou_threshold
        unmatched_dets = np.concatenate([unmatched_dets, matched_indices[low, 0]])
        unmatched_trks = np.concatenate([unmatched_trks, matched_indices[low, 1]])
        matches = matched_indices[~low]
    else:
        unmatched_dets = np.arange(num_dets)
        unmatched_trks = np.arange(num_trks)
        matches = np.empty((0, 2), dtype=int)
    return matches, unmatched_dets.astype(int), unmatched_trks.astype(int)


def associate_ocm(
    detections: np.ndarray,
    predicted: np.ndarray,
    iou_threshold: float,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    inertia: float,
    similarity=iou_batch,
):
    """Stage 1: IoU with the motion-model predictions + OCM direction-consistency term.

    cost = similarity(det, predicted_box) + inertia * angle_term * det_score, where
    angle_term in [-0.5, 0.5] rewards detections lying in the direction the track
    has been moving (direction estimated from observations, not from the filter).
    The official code always uses plain IoU here (``asso_func`` only affects the
    BYTE and OCR rounds), so the tracker calls this with the default.
    """
    if len(predicted) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y = velocities[:, 0][:, np.newaxis]
    inertia_X = velocities[:, 1][:, np.newaxis]
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[previous_obs[:, 4] < 0] = 0  # track has no previous observation

    iou_matrix = similarity(detections, predicted)
    scores = detections[:, -1][:, np.newaxis]
    angle_diff_cost = ((valid_mask[:, np.newaxis] * diff_angle) * inertia).T * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    return _filter_matches(matched_indices, iou_matrix, iou_threshold, len(detections), len(predicted))


def associate_iou(
    detections: np.ndarray,
    track_boxes: np.ndarray,
    iou_threshold: float,
    similarity=iou_batch,
) -> list[tuple[int, int]] | None:
    """Hungarian matching on similarity alone, used by the BYTE and OCR rounds.

    Returns ``(det_index, track_index)`` pairs local to the given arrays, or
    ``None`` when no pair clears the threshold and the assignment is skipped.
    The distinction matters for exact parity: the official code only rewrites
    (and thereby sorts) its unmatched index lists when the assignment ran.
    """
    iou_left = np.array(similarity(detections, track_boxes))
    if iou_left.max() <= iou_threshold:
        return None
    pairs = []
    for m in linear_assignment(-iou_left):
        if iou_left[m[0], m[1]] < iou_threshold:
            continue
        pairs.append((int(m[0]), int(m[1])))
    return pairs
