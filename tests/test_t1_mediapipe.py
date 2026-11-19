"""MediaPipe eye-direction cues and the VFOA fallback for rows the mesh could not read."""

import numpy as np
import pytest

from grouppulse.perception.backends import MESH_6, eye_direction
from grouppulse.perception.io import TrackRow, read_tracks, write_tracks
from grouppulse.perception.vfoa import FEATURE_SETS, Sample, VfoaModel, fit


def test_eye_direction_signs():
    # Person looks to their own left = toward image right: left eye looks out, right eye looks in.
    h, v = eye_direction({"eyeLookOutLeft": 0.6, "eyeLookInRight": 0.6})
    assert h == pytest.approx(0.6) and v == 0
    h, v = eye_direction({"eyeLookInLeft": 0.4, "eyeLookOutRight": 0.4})
    assert h == pytest.approx(-0.4)
    _, v = eye_direction({"eyeLookDownLeft": 0.7, "eyeLookDownRight": 0.5})
    assert v == pytest.approx(-0.6)
    assert eye_direction({}) == (0.0, 0.0)


def test_mesh_indices_match_model_order():
    # eye corners, nose, mouth corners, chin: six distinct face-mesh points
    assert len(set(MESH_6)) == 6 and MESH_6[2] == 1 and MESH_6[5] == 152


def _eyes_data(seed):
    """P3 and table share head yaw/pitch; only eye_v (looking down) separates them."""
    rng = np.random.default_rng(seed)
    out = []
    for t, ev in (("P3", 0.0), ("table", -0.5)):
        out += [Sample("Closeup1", rng.normal(0, 5), rng.normal(3, 2), t, (rng.normal(0, .1), rng.normal(ev, .1)))
                for _ in range(80)]
    # rows where the mesh failed (profile faces): yaw/pitch only
    out += [Sample("Closeup1", rng.normal(70, 4), rng.normal(0, 2), "P2") for _ in range(60)]
    return out


def test_eyes_model_uses_fallback_for_rows_without_mesh():
    feats = FEATURE_SETS["eyes"]
    model = fit(_eyes_data(0), "e", features=feats)
    assert model.features == feats
    assert "P2" not in model.cameras["Closeup1"]          # never seen with eye cues
    assert "P2" in model.fallback["Closeup1"]
    test = _eyes_data(1)
    acc = np.mean([model.predict(s.view, s.yaw, s.pitch, s.extra) == s.target for s in test])
    assert acc > 0.9
    assert model.predict("Closeup1", 70, 0) == "P2"        # no eye cues -> fallback


def test_fallback_survives_save_load(tmp_path):
    model = fit(_eyes_data(0), "e", features=FEATURE_SETS["eyes"])
    again = VfoaModel.load(model.save(tmp_path / "m.toml"))
    assert again.features == FEATURE_SETS["eyes"]
    assert set(again.fallback["Closeup1"]) == set(model.fallback["Closeup1"])
    assert again.predict("Closeup1", 0, 3, (0.0, -0.5)) == "table"


def test_track_rows_keep_eye_columns(tmp_path):
    r = TrackRow("ES2008a", 5, 0.2, "P1", True, 10.0, -2.0, None, "P3", None, None, 0.0, False,
                 eye_h=0.12, eye_v=-0.4, face_model="hybrid")
    assert read_tracks(write_tracks([r], tmp_path / "t.csv")) == [r]
