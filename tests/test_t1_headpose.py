"""T1.3 logic: head yaw/pitch recovered from landmarks within 10 degrees (synthetic faces)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from grouppulse.perception import head_pose, project_model  # noqa: E402

POSES = [(y, p) for y in (-60, -40, -20, 0, 20, 40, 60) for p in (-30, -15, 0, 15)]


@pytest.mark.parametrize("yaw,pitch", POSES)
def test_six_point_exact(yaw, pitch):
    est = head_pose(project_model(yaw, pitch), 640, 480)
    assert est is not None
    assert est[0] == pytest.approx(yaw, abs=1.0)
    assert est[1] == pytest.approx(pitch, abs=1.0)


def test_sign_convention():
    # +yaw = facing image right: the nose moves right of the eye midpoint
    pts = project_model(30, 0)
    assert pts[2, 0] > (pts[0, 0] + pts[1, 0]) / 2
    # +pitch = looking up: the nose moves up (smaller y) relative to the mouth corners
    up, flat = project_model(0, 20), project_model(0, 0)
    assert (up[3, 1] - up[2, 1]) > (flat[3, 1] - flat[2, 1])


def test_five_point_with_pixel_noise_meets_t13():
    rng = np.random.default_rng(0)
    errs = []
    for yaw, pitch in POSES:
        if abs(yaw) > 40:
            continue  # YuNet-style 5 points get unstable past ~45 degrees; the 6-point path covers that
        pts = project_model(yaw, pitch, points=5) + rng.normal(0, 1.5, (5, 2))
        est = head_pose(pts, 640, 480)
        assert est is not None
        errs.append(max(abs(est[0] - yaw), abs(est[1] - pitch)))
    assert np.mean(errs) <= 10.0


def test_bad_input_returns_none():
    assert head_pose(np.zeros((3, 2)), 640, 480) is None
    assert head_pose(np.full((5, 2), np.nan), 640, 480) is None
