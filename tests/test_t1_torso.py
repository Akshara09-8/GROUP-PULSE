"""T1.4 logic: 8-way torso direction from shoulder keypoints; T1.6 partly hidden people."""

import math

import numpy as np
import pytest

from grouppulse.perception import bin8, frontal_width, torso_yaw
from grouppulse.perception.metrics import bin8_accuracy
from grouppulse.perception.torso import L_SH, NOSE, R_SH, L_EAR, R_EAR

W = 120.0  # frontal shoulder width in pixels


def skeleton(yaw: float, noise: float = 0.0, rng=None, nose_visible: bool = True) -> np.ndarray:
    """Orthographic projection of a seated upper body turned by `yaw` (camera-relative)."""
    rng = rng or np.random.default_rng(0)
    a = math.radians(yaw)
    kps = np.zeros((17, 3))
    cx, cy = 320.0, 300.0
    # Person's left shoulder sits at +W/2 along the body's left axis. Facing the camera, that is image right.
    kps[L_SH] = [cx + (W / 2) * math.cos(a), cy, 0.9]
    kps[R_SH] = [cx - (W / 2) * math.cos(a), cy, 0.9]
    # Nose 25 px in front of the neck along the facing direction; its image x shifts by sin(yaw).
    kps[NOSE] = [cx + 25 * math.sin(a), cy - 70, 0.9 if nose_visible and math.cos(a) > -0.2 else 0.0]
    kps[L_EAR] = [cx + 30 * math.cos(a) - 10 * math.sin(a), cy - 75, 0.8]
    kps[R_EAR] = [cx - 30 * math.cos(a) - 10 * math.sin(a), cy - 75, 0.8]
    kps[:, :2] += rng.normal(0, noise, (17, 2))
    return kps


def test_eight_directions_with_noise_meet_t14():
    rng = np.random.default_rng(1)
    true, pred = [], []
    for yaw in range(-180, 180, 5):
        for _ in range(4):
            y, _ = torso_yaw(skeleton(yaw, noise=3.0, rng=rng), W)
            true.append(yaw)
            pred.append(y)
    assert bin8_accuracy(true, pred) >= 0.80


@pytest.mark.parametrize("yaw", [0, 45, 90, -90, 135, 180])
def test_clean_bins(yaw):
    y, conf = torso_yaw(skeleton(yaw), W)
    assert bin8(y) == bin8(yaw)
    assert conf > 0.4


def test_hidden_shoulder_returns_none():
    kps = skeleton(0)
    kps[R_SH, 2] = 0.05
    assert torso_yaw(kps, W) == (None, 0.0)


def test_no_head_cues_lowers_confidence_but_keeps_frame():
    kps = skeleton(60)
    kps[[NOSE, L_EAR, R_EAR], 2] = 0.0
    y, conf = torso_yaw(kps, W)
    assert y is not None and conf < 0.5


def test_frontal_width_from_track():
    widths = [W * math.cos(math.radians(a)) for a in range(-80, 81, 10)] + [None, None]
    assert frontal_width(widths) == pytest.approx(W, rel=0.03)
    assert frontal_width([None, 10.0]) is None
