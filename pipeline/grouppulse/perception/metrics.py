"""Phase 1 test metrics (T1.1-T1.5) computed against hand labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from .geometry import angle_diff, bin8
from .tracker import Box, iou


def detection_recall(gt: Mapping[int, list[Box]], pred: Mapping[int, list[Box]], thr: float = 0.5) -> float:
    """T1.1: share of ground-truth boxes (per frame) matched by a prediction with IoU >= thr."""
    hit = total = 0
    for frame, gboxes in gt.items():
        pboxes = list(pred.get(frame, []))
        for g in gboxes:
            total += 1
            best = max(range(len(pboxes)), key=lambda i: iou(g, pboxes[i]), default=None)
            if best is not None and iou(g, pboxes[best]) >= thr:
                hit += 1
                pboxes.pop(best)
    return hit / total if total else 1.0


def id_switches(gt_to_pred: Iterable[tuple[int, int, int | None]]) -> int:
    """T1.2: count identity switches from (frame, gt_id, matched_pred_id) rows."""
    last: dict[int, int] = {}
    switches = 0
    for _, g, p in sorted(gt_to_pred):
        if p is None:
            continue
        if g in last and last[g] != p:
            switches += 1
        last[g] = p
    return switches


def mean_angle_error(true: Iterable[float], pred: Iterable[float | None]) -> float:
    """T1.3: mean absolute angular error, ignoring frames with no prediction."""
    errs = [angle_diff(t, p) for t, p in zip(true, pred) if p is not None]
    return float(np.mean(errs)) if errs else float("nan")


def bin8_accuracy(true_yaw: Iterable[float], pred_yaw: Iterable[float | None]) -> float:
    """T1.4: share of frames where the predicted 8-way direction bin is correct."""
    pairs = [(bin8(t), bin8(p)) for t, p in zip(true_yaw, pred_yaw) if p is not None]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


def target_accuracy(true: Iterable[str], pred: Iterable[str]) -> float:
    """T1.5: frame accuracy of gaze targets."""
    pairs = list(zip(true, pred))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


def confusion(true: Iterable[str], pred: Iterable[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in zip(true, pred):
        out[a][b] += 1
    return {k: dict(v) for k, v in out.items()}
