"""Adjusted counting: correct per-camera gaze-target confusion when measuring *how much* attention
each target gets (a quantification problem), not which target one frame shows.

Why: the per-frame gaze model confuses some targets with others in a consistent, camera-specific
way. On the Idiap meetings, glances at P4 were often read as P2, so P2's attention received was
over-estimated in 9 of 11 held-out meetings. Counting the raw predicted labels carries that bias
straight into dominance. Adjusted counting (Forman 2008; Saerens et al. 2002) estimates each
camera's confusion matrix C[true, predicted] on held-out training data and, for a new session,
solves for the true target mix q that best explains the predicted mix p: p ≈ Cᵀ q, q ≥ 0, Σq = 1.

On 11 held-out Idiap meetings this lifted dominance rank correlation from +0.18 to +0.44 and
top-1 from 2/11 to 7/11 without changing the gaze model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

TARGETS = ("P1", "P2", "P3", "P4", "P5", "P6", "screen", "table", "whiteboard", "away")


def confusion_matrices(pairs: Iterable[tuple[str, str, str]], targets=TARGETS) -> dict[str, np.ndarray]:
    """(view, true target, predicted target) triples -> row-normalised C[true, predicted] per view.

    Rows for targets never seen as truth are left as identity, so the correction does nothing there."""
    idx = {t: i for i, t in enumerate(targets)}
    raw: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((len(targets), len(targets))))
    for view, t, p in pairs:
        if t in idx and p in idx:
            raw[view][idx[t], idx[p]] += 1
    out = {}
    for view, m in raw.items():
        c = np.eye(len(targets))
        seen = m.sum(1) > 0
        c[seen] = m[seen] / m[seen].sum(1, keepdims=True)
        out[view] = c
    return out


def adjust_distribution(p: np.ndarray, c: np.ndarray, iters: int = 2000) -> np.ndarray:
    """Least-squares true distribution q with Cᵀ q ≈ p on the probability simplex (projected gradient)."""
    a = c.T
    step = 1.0 / max(np.linalg.norm(a, 2) ** 2, 1e-9)
    q = np.full(len(p), 1.0 / len(p))
    for _ in range(iters):
        q = q - step * (a.T @ (a @ q - p))
        q = _project_simplex(q)
    return q


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {q >= 0, sum q = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0)


def adjusted_shares(labels: list[str], c: np.ndarray, targets=TARGETS) -> dict[str, float]:
    """One viewer's predicted labels (unknown dropped) -> corrected share of time on each target."""
    idx = {t: i for i, t in enumerate(targets)}
    known = [x for x in labels if x in idx]
    if not known:
        return {}
    p = np.zeros(len(targets))
    for x in known:
        p[idx[x]] += 1
    p /= p.sum()
    q = adjust_distribution(p, c)
    return {t: float(q[i]) for i, t in enumerate(targets)}


def to_jsonable(cs: dict[str, np.ndarray]) -> dict:
    return {"targets": list(TARGETS), "matrices": {v: np.round(m, 5).tolist() for v, m in cs.items()}}


def from_jsonable(d: dict | None) -> dict[str, np.ndarray]:
    if not d:
        return {}
    if list(d.get("targets", TARGETS)) != list(TARGETS):
        raise ValueError("confusion matrices were saved with a different target list")
    return {v: np.asarray(m, float) for v, m in d["matrices"].items()}
