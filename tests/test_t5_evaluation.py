"""Phase 5 helpers: bootstrap CIs (T5.5), determinism (T5.4), window-level F1 (T5.2), report verdicts."""

import math

import pytest

from grouppulse.evaluation.report import build
from grouppulse.evaluation.run import pooled_f1, rank_score
from grouppulse.evaluation.stats import bootstrap_ci, mean, prf, window_labels


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    xs = [0.40, 0.55, 0.60, 0.70, 0.75, 0.52, 0.66, 0.58]
    a = bootstrap_ci(xs, mean)
    b = bootstrap_ci(xs, mean)
    assert a == b                                        # T5.4: same seed, same interval
    point, lo, hi = a
    assert point == pytest.approx(mean(xs))
    assert lo < point < hi and 0.4 <= lo and hi <= 0.75
    assert bootstrap_ci(xs, mean, seed=14) != a          # the seed really drives it


def test_bootstrap_empty_and_constant():
    assert all(math.isnan(v) for v in bootstrap_ci([], mean))
    assert bootstrap_ci([0.5] * 5, mean) == (0.5, 0.5, 0.5)


def test_window_labels_half_window_rule():
    spells = [{"person": "P1", "start": 12.0, "end": 31.0}]
    lab = window_labels(spells, ["P1", "P2"], end=40.0, window_s=10.0)
    assert [lab[(i, "P1")] for i in range(4)] == [False, True, True, False]   # 10-20 and 20-30 covered; 30-40 only 1 s
    assert not any(lab[(i, "P2")] for i in range(4))


def test_prf():
    t = {1: True, 2: True, 3: False, 4: False}
    p = {1: True, 2: False, 3: True, 4: False}
    r = prf(t, p)
    assert (r["precision"], r["recall"], r["f1"]) == (0.5, 0.5, 0.5)
    assert prf({1: False}, {1: False})["f1"] == 0.0


def test_pooled_f1_keeps_meetings_apart():
    a = {"meeting": "A", "truth": {(0, "P1"): True}, "pred": {(0, "P1"): True}}
    b = {"meeting": "B", "truth": {(0, "P1"): False}, "pred": {(0, "P1"): True}}
    assert pooled_f1([a, b]) == pytest.approx(2 * 0.5 * 1 / 1.5)


def test_rank_score():
    s = rank_score(["P2", "P1", "P3", "P4"], ["P2", "P3", "P1", "P4"])
    assert s["top1"] is True and s["rho"] == pytest.approx(0.8)


def test_report_has_verdicts_and_honest_fails():
    res = {"settings": {"bootstrap": "2000 resamples of meetings, seed 13"},
           "E1_gaze_idiap": {"series": ["IS1000"], "per_meeting": [{"meeting": "IS1000a", "accuracy": 0.6,
                             "majority_baseline": 0.2, "labelled_frames": 10}],
                             "mean_accuracy": {"value": 0.6, "ci95": [0.5, 0.7], "n": 1},
                             "mean_majority_baseline": {"value": 0.2, "ci95": [0.2, 0.2], "n": 1}},
           "E4_disengagement_idiap": {"note": "proxy", "per_meeting": [], "pooled_f1": {"value": 0.65, "ci95": [0.5, 0.8], "n": 3}}}
    md = build(res, repro=True)
    assert "| T1.5 gaze target ≥ 70% | **Fail** |" in md
    assert "| T5.2 disengagement F1 ≥ 0.6 (proxy truth) | Pass |" in md
    assert "| T5.4 reproducible (two runs, identical output) | Pass |" in md
    assert md.index("## Verdicts") < md.index("## E1")
