"""Adjusted counting (quantify.py): undo a camera's consistent gaze-target confusion."""

import numpy as np
import pytest

from grouppulse.perception import load_layout
from grouppulse.perception.io import TrackRow
from grouppulse.perception.quantify import (TARGETS, adjust_distribution, adjusted_shares, confusion_matrices,
                                            from_jsonable, to_jsonable)
from grouppulse.perception.vfoa import Sample, VfoaModel, fit
from grouppulse.social import analyze

I = {t: i for i, t in enumerate(TARGETS)}


def leaky(source="P4", into="P2", rate=0.5):
    """Confusion where `rate` of glances at `source` are read as `into` (the Idiap P4 -> P2 pattern)."""
    c = np.eye(len(TARGETS))
    c[I[source], I[source]] = 1 - rate
    c[I[source], I[into]] = rate
    return c


def test_adjust_recovers_true_mix():
    c = leaky()
    q_true = np.zeros(len(TARGETS))
    q_true[[I["P2"], I["P4"], I["table"]]] = [0.2, 0.3, 0.5]
    p = c.T @ q_true                                          # what the camera reports: P2 0.35, P4 0.15
    assert p[I["P2"]] == pytest.approx(0.35)
    q = adjust_distribution(p, c)
    assert q == pytest.approx(q_true, abs=1e-3)
    assert q.sum() == pytest.approx(1.0) and (q >= 0).all()


def test_identity_confusion_changes_nothing():
    labels = ["P2"] * 3 + ["table"] * 7
    s = adjusted_shares(labels, np.eye(len(TARGETS)))
    assert s["P2"] == pytest.approx(0.3, abs=1e-3) and s["table"] == pytest.approx(0.7, abs=1e-3)


def test_confusion_from_pairs_and_round_trip():
    pairs = [("Closeup1", "P4", "P2")] * 2 + [("Closeup1", "P4", "P4")] * 2 + [("Closeup1", "table", "table")] * 3
    cs = confusion_matrices(pairs)
    c = cs["Closeup1"]
    assert c[I["P4"], I["P2"]] == pytest.approx(0.5) and c[I["table"], I["table"]] == 1.0
    assert c[I["P1"], I["P1"]] == 1.0                         # unseen truth rows stay identity
    back = from_jsonable(to_jsonable(cs))
    assert np.allclose(back["Closeup1"], c, atol=1e-5)


def test_analyze_with_confusion_undoes_p2_inflation():
    room = load_layout("idiap")
    # True behaviour: P1 and P3 look at P4 and P2 equally; their cameras read half the P4 glances as P2.
    rows = []
    for k in range(400):
        t = k / 5
        p4_glance = k % 2 == 0
        for viewer in ("P1", "P3"):
            seen = "P4" if p4_glance and k % 4 == 0 else "P2"      # 50% of P4 glances -> P2
            rows.append(TrackRow("IS0000a", k * 5, t, viewer, True, 0.0, 0.0, None, seen, None, None, 0.0, False))
        rows.append(TrackRow("IS0000a", k * 5, t, "P2", True, 0.0, 0.0, None, "table", None, None, 0.0, False))
        rows.append(TrackRow("IS0000a", k * 5, t, "P4", True, 0.0, 0.0, None, "table", None, None, 0.0, False))
    raw = analyze(rows, room)
    assert raw["group"]["dominant"] == "P2" and raw["group"]["attention_method"] == "raw"
    conf = {room.camera_of[v]: leaky() for v in ("P1", "P3")}
    adj = analyze(rows, room, confusion=conf)
    a = adj["per_person"]
    assert adj["group"]["attention_method"] == "adjusted"
    assert a["P2"]["attention_received"] == pytest.approx(a["P4"]["attention_received"], abs=0.02)
    assert adj["group"]["attention_raw"]["P2"] > a["P2"]["attention_received"]


def test_model_saves_and_loads_confusion(tmp_path):
    m = fit([Sample("Closeup1", d, 0, "P2") for d in range(-5, 6)] * 3, "idiap", min_count=5)
    m.confusion = {"Closeup1": leaky()}
    back = VfoaModel.load(m.save(tmp_path / "v.toml"))
    assert np.allclose(back.confusion["Closeup1"], leaky(), atol=1e-5)
    plain = fit([Sample("Closeup1", d, 0, "P2") for d in range(-5, 6)] * 3, "idiap", min_count=5)
    assert VfoaModel.load(plain.save(tmp_path / "p.toml")).confusion == {}
