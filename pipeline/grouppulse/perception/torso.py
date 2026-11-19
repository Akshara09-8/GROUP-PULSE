"""Body (torso) orientation from 2D COCO pose keypoints.

Seated people at a meeting table usually have their hips hidden, so this uses shoulders only:
- cos(yaw) ~ signed shoulder width / the person's frontal shoulder width
  (the sign flips when the person turns their back, because left/right shoulders swap sides),
- the side (left or right) comes from where the nose or ears sit relative to the shoulder midpoint.

The frontal width is learned per track (a high percentile of observed widths), so no body-size
assumption is needed.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6


def _ok(kps: np.ndarray, i: int, thr: float) -> bool:
    return kps.shape[1] < 3 or kps[i, 2] >= thr


def signed_shoulder_width(kps: np.ndarray, thr: float = 0.3) -> float | None:
    """Person's-left shoulder x minus person's-right shoulder x. Positive = chest toward camera."""
    kps = np.asarray(kps, dtype=float)
    if not (_ok(kps, L_SH, thr) and _ok(kps, R_SH, thr)):
        return None
    return float(kps[L_SH, 0] - kps[R_SH, 0])


def frontal_width(widths: Iterable[float | None], q: float = 95.0) -> float | None:
    """Estimate a track's frontal shoulder width from many frames."""
    w = [abs(x) for x in widths if x is not None]
    return float(np.percentile(w, q)) if len(w) >= 5 else None


def torso_yaw(kps: np.ndarray, ref_width: float, thr: float = 0.3) -> tuple[float | None, float]:
    """(yaw, confidence). yaw is camera-relative (0 = facing camera, +90 = facing image right).

    Returns (None, 0.0) when the shoulders are not visible. Confidence is lowered, not zeroed,
    for partly hidden people (test T1.6) so the frame is kept and marked."""
    kps = np.asarray(kps, dtype=float)
    w = signed_shoulder_width(kps, thr)
    if w is None or not ref_width:
        return None, 0.0
    c = max(-1.0, min(1.0, w / ref_width))
    mid_x = (kps[L_SH, 0] + kps[R_SH, 0]) / 2.0
    # Side cue: nose (or the mean of visible ears) relative to the shoulder midpoint.
    side_pts = [kps[i, 0] for i in (NOSE,) if _ok(kps, i, thr)]
    if not side_pts:
        side_pts = [kps[i, 0] for i in (L_EAR, R_EAR) if _ok(kps, i, thr)]
    offset = (np.mean(side_pts) - mid_x) if side_pts else 0.0
    side = 1.0 if offset >= 0 else -1.0
    if c < 0:  # back to camera: the head offset points away from the facing side
        side = -side
    yaw = math.degrees(math.atan2(side * math.sqrt(max(0.0, 1 - c * c)), c))
    conf = float(np.mean([kps[i, 2] for i in (L_SH, R_SH)])) if kps.shape[1] >= 3 else 1.0
    if not side_pts:
        conf *= 0.5  # side is a guess
    return yaw, conf
