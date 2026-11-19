"""Close-up torso step: readings at 1 fps are held across the 5 fps rows; missing people stay None."""

import math

import numpy as np

from grouppulse.perception import run as runmod
from grouppulse.perception.backends import PersonPose
from grouppulse.perception.io import TrackRow
from grouppulse.perception.torso import L_SH, NOSE, R_SH


def kps(yaw, w=120.0):
    a = math.radians(yaw)
    k = np.zeros((17, 3))
    k[L_SH] = [200 + w / 2 * math.cos(a), 250, 0.9]
    k[R_SH] = [200 - w / 2 * math.cos(a), 250, 0.9]
    k[NOSE] = [200 + 25 * math.sin(a), 180, 0.9]
    return k


class FakePose:
    """Turns 40 degrees to image right from frame 50 on; nobody in frames 100-124."""

    def largest(self, frame):
        i = int(frame)
        if 100 <= i < 125:
            return None
        return PersonPose(None, (0, 0, 400, 300), kps(40 if i >= 50 else 0), 0.9)


def test_hold_and_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(runmod, "_frames",
                        lambda path, every, limit: ((i, i) for i in range(0, 150, every)))
    monkeypatch.setattr(runmod, "video_path", lambda data, m, v: tmp_path)   # exists
    rows = {(i, p): TrackRow("ES2002a", i, i / 25, p, True, 0.0, 0.0, None, "P3", None, None, 0.0, False)
            for i in range(0, 150, 5) for p in ("P1", "P2", "P3", "P4")}
    # frontal width needs >= 5 readings: 6 samples per camera at 25-frame steps
    runmod.add_torso_closeups(rows, tmp_path, "ES2002a", every=5, limit=None, pose=FakePose(), torso_every=25)
    p1 = {i: rows[(i, "P1")] for i in range(0, 150, 5)}
    assert all(abs(p1[i].torso_yaw) < 5 for i in range(0, 50, 5))          # held from frames 0 and 25
    assert all(30 < p1[i].torso_yaw < 50 for i in range(50, 100, 5))       # held from frames 50 and 75
    assert all(p1[i].torso_yaw is None and p1[i].low_confidence for i in range(100, 125, 5))
    assert p1[0].torso_dir is not None and p1[100].torso_dir is None
