"""Small, dependency-free statistics for the evaluation: bootstrap CIs and window-level F1."""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def bootstrap_ci(items: Sequence[T], stat: Callable[[Sequence[T]], float], n: int = 2000,
                 seed: int = 13, level: float = 0.95) -> tuple[float, float, float]:
    """(point estimate, low, high): percentile bootstrap, resampling whole items (meetings).

    Meetings, not frames, are resampled because frames within a meeting are strongly correlated;
    resampling frames would give falsely narrow intervals. Fixed seed -> identical reruns (T5.4)."""
    if not items:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(items)
    vals = sorted(stat([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    lo = vals[int((1 - level) / 2 * n)]
    hi = vals[min(n - 1, int((1 + level) / 2 * n))]
    return stat(items), lo, hi


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def window_labels(spells: list[dict], people: list[str], end: float, window_s: float) -> dict[tuple[int, str], bool]:
    """Mark each (window, person) disengaged if at least half the window lies inside one of their spells."""
    out = {}
    n = int(end // window_s) + (1 if end % window_s else 0)
    for p in people:
        mine = [s for s in spells if s["person"] == p]
        for i in range(n):
            a, b = i * window_s, min(end, (i + 1) * window_s)
            inside = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for s in mine)
            out[(i, p)] = inside >= 0.5 * (b - a)
    return out


def prf(truth: dict, pred: dict) -> dict[str, float]:
    """Precision, recall, F1 over shared keys (positive = disengaged)."""
    tp = sum(1 for k, t in truth.items() if t and pred.get(k))
    fp = sum(1 for k, p in pred.items() if p and not truth.get(k))
    fn = sum(1 for k, t in truth.items() if t and not pred.get(k))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
