"""Learned VFOA and geometric camera calibration on synthetic glances."""

import numpy as np
import pytest

from grouppulse.perception import load_layout, wrap180
from grouppulse.perception.geometry import direction_to
from grouppulse.perception.vfoa import Sample, VfoaModel, fit, fit_camera_dirs, smooth

CENTRES = {"P2": (-35, 0), "P3": (5, 2), "P4": (40, 0), "screen": (-60, 5), "table": (0, -30)}


def synth(view="Closeup1", n=60, sd=6, seed=0):
    rng = np.random.default_rng(seed)
    return [Sample(view, cy + rng.normal(0, sd), cp + rng.normal(0, sd), t)
            for t, (cy, cp) in CENTRES.items() for _ in range(n)]


def test_fit_and_predict():
    model = fit(synth(seed=1), "edinburgh")
    test = synth(seed=2)
    acc = np.mean([model.predict(s.view, s.yaw, s.pitch) == s.target for s in test])
    assert acc > 0.85
    assert model.predict("Closeup9", 0, 0) == "unknown"
    assert model.predict("Closeup1", None, 0) == "unknown"


def test_rare_targets_are_dropped():
    samples = synth() + [Sample("Closeup1", 80, 0, "whiteboard")] * 3
    assert "whiteboard" not in fit(samples, "x").cameras["Closeup1"]


def test_save_load_round_trip(tmp_path):
    model = fit(synth(), "edinburgh")
    again = VfoaModel.load(model.save(tmp_path / "v.toml"))
    assert again.site == "edinburgh"
    a, b = model.cameras["Closeup1"]["P3"], again.cameras["Closeup1"]["P3"]
    assert a.mean == pytest.approx(b.mean, abs=1e-3) and a.prior == pytest.approx(b.prior, abs=1e-4)
    assert again.predict("Closeup1", 5, 2) == "P3"


def test_extra_cues_separate_what_angles_cannot():
    # Same yaw/pitch for "P3" and "table"; only the nose-drop cue tells them apart (the real ES2008a problem).
    rng = np.random.default_rng(4)
    mk = lambda t, dy, n=80: [Sample("Closeup1", rng.normal(0, 5), rng.normal(3, 2), t,  # noqa: E731
                                     (0.0, rng.normal(dy, 0.04), 0.5, 0.5, 0.2)) for _ in range(n)]
    train, test = mk("P3", 0.45) + mk("table", 0.70), mk("P3", 0.45) + mk("table", 0.70)
    with_cues = fit(train, "e")
    angles = fit([Sample(s.view, s.yaw, s.pitch, s.target) for s in train], "e")
    acc = lambda m, use: np.mean([m.predict(s.view, s.yaw, s.pitch, s.extra if use else ()) == s.target  # noqa: E731
                                  for s in test])
    assert acc(with_cues, True) > 0.95
    assert acc(angles, False) < 0.7


def test_smooth_removes_single_frame_flicker():
    assert smooth(["P2", "P2", "P3", "P2", "P2"], 5) == ["P2"] * 5
    assert smooth(["P2", "P3"], 1) == ["P2", "P3"]


def test_viterbi_removes_flicker_and_keeps_real_switches():
    from grouppulse.perception.vfoa import viterbi

    a, b = {"P2": 0.0, "P3": -3.0}, {"P2": -3.0, "P3": 0.0}
    frames = [a] * 10 + [b] + [a] * 10 + [b] * 10        # one-frame blip, then a real switch
    path = viterbi(frames, stay=0.9)
    assert path[10] == "P2"                              # blip smoothed away
    assert path[-1] == "P3" and path[25] == "P3"         # sustained switch kept
    assert viterbi([a, {}, a])[1] == "unknown"           # no face stays unknown
    assert viterbi([]) == [] and viterbi([{}]) == ["unknown"]


def test_fit_camera_dirs_recovers_true_direction():
    layout = load_layout("edinburgh")
    true_dir = -60.0  # the real camera is angled, not -90 as configured
    samples = []
    for tgt in ("P2", "P3", "P4", "screen"):
        pos = layout.seats.get(tgt) or layout.objects[tgt]
        want = direction_to(layout.seats["P1"], pos)
        yaw = wrap180(want - true_dir - 180)
        samples += [Sample("Closeup1", yaw + d, 0, tgt) for d in (-3, 0, 2)]
    assert fit_camera_dirs(samples, layout)["Closeup1"] == pytest.approx(true_dir, abs=3.5)
