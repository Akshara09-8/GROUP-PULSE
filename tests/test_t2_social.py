"""Phase 2 tests T2.1-T2.8 on synthetic meetings with known answers."""

import math

import pytest

from grouppulse.perception import load_layout
from grouppulse.perception.geometry import direction_to
from grouppulse.perception.io import TrackRow
from grouppulse.social import SocialConfig, analyze
from grouppulse.social.report import spearman, with_true_gaze
from grouppulse.social.metrics import (Series, attention_received, gaze_following_rate, gaze_shares,
                                       mutual_gaze_rate)

FPS = 5
ROOM = load_layout("edinburgh")


def meeting(targets, seconds=120, torso=None, people=("P1", "P2", "P3", "P4")):
    """targets: person -> function(t) -> gaze target. torso: person -> function(t) -> (yaw, dir)."""
    rows = []
    for k in range(int(seconds * FPS)):
        t = k / FPS
        for p in people:
            yaw, tdir = (torso[p](t) if torso and p in torso else (0.0, None))
            rows.append(TrackRow("SYN", k * 5, t, p, True, 0.0, 0.0, None, targets[p](t), None,
                                 yaw, 0.9, False, torso_dir=tdir))
    return rows


def const(x):
    return lambda t: x


def test_t21_attention_shares_add_up():
    rows = meeting({"P1": const("P2"), "P2": lambda t: "P1" if t < 60 else "table",
                    "P3": lambda t: "P1" if t < 60 else "screen", "P4": lambda t: "P1" if t < 60 else "away"})
    s = Series.from_rows(rows)
    att = attention_received(s)
    assert att["P1"] == pytest.approx(0.5)       # P2-P4 look at P1 half the time
    assert att["P2"] == pytest.approx(1 / 3)     # only P1 looks at P2, all the time
    assert att["P3"] == att["P4"] == 0
    for shares in gaze_shares(s).values():
        assert sum(shares.values()) == pytest.approx(1.0)


def test_t22_most_looked_at_is_dominant():
    rows = meeting({"P1": const("P3"), "P2": const("P3"), "P3": const("P1"), "P4": lambda t: "P3" if t % 10 < 7 else "P1"})
    summ = analyze(rows, ROOM, SocialConfig())
    assert summ["group"]["dominant"] == "P3"
    assert summ["group"]["dominance_rank"][0] == "P3"


def test_t23_long_look_away_is_flagged_short_is_not():
    rows = meeting({"P1": const("P2"), "P2": const("P1"), "P3": lambda t: "away" if 20 <= t < 40 else "P1",
                    "P4": lambda t: "away" if 50 <= t < 90 else "P1"})
    spells = analyze(rows, ROOM, SocialConfig())["disengaged_spells"]
    assert [x["person"] for x in spells] == ["P4"]            # P3's 20 s is under the 30 s minimum
    assert 35 <= spells[0]["end"] - spells[0]["start"] <= 45  # about 40 s
    assert spells[0]["start"] == pytest.approx(50, abs=3)


def test_t24_rapport_higher_with_mutual_gaze_and_following():
    lively = meeting({"P1": lambda t: "P2" if int(t) % 4 < 2 else "screen",
                      "P2": lambda t: "P1" if int(t) % 4 < 2 else ("screen" if (t % 4) >= 2.6 else "P1"),
                      "P3": lambda t: "P4" if int(t) % 4 < 2 else ("screen" if (t % 4) >= 3.0 else "P4"),
                      "P4": lambda t: "P3"})
    flat = meeting({p: const("table") for p in ("P1", "P2", "P3", "P4")})
    a = analyze(lively, ROOM, SocialConfig())["group"]
    b = analyze(flat, ROOM, SocialConfig())["group"]
    # P1-P2 and P3-P4 look at each other for part of each 4 s cycle: 2 of 6 pairs, ~60% of the time
    assert a["mutual_gaze_rate"] > 0.15 and b["mutual_gaze_rate"] == 0
    assert a["gaze_following_rate"] > 0.2
    assert a["rapport_index"] > b["rapport_index"]


def test_gaze_following_counts_copies_within_window():
    rows = meeting({"P1": lambda t: "screen" if t >= 10 else "P2", "P2": lambda t: "screen" if t >= 11 else "P1",
                    "P3": lambda t: "screen" if t >= 20 else "P1", "P4": const("P1")}, seconds=30)
    s = Series.from_rows(rows)
    # shifts: P1->P2(0), P1->screen(10), P2->P1(0), P2->screen(11), P3->P1(0), P3->screen(20), P4->P1(0)
    # copied within 2 s: P1->screen (P2 at 11); P2->P1 at 0 by P3/P4 not *after* -> no; so 1 of 7
    assert gaze_following_rate(s, 2.0) == pytest.approx(1 / 7)


def test_t25_addressee_rules():
    rows = meeting({"P1": const("P2"), "P2": const("P1"), "P3": const("P2"), "P4": const("away")}, seconds=20)
    w = analyze(rows, ROOM, SocialConfig())["windows"][0]
    assert w["addressee"] == {"person": "P4", "reason": "re-engage least engaged"}
    w = analyze(rows, ROOM, SocialConfig(addressee_rule="dominant"))["windows"][0]
    assert w["addressee"]["person"] in ("P1", "P2") and w["addressee"]["reason"] == "address dominant"


def test_t26_window_size_keeps_scores_stable_and_finite():
    rows = meeting({"P1": lambda t: "P2" if int(t) % 3 else "table", "P2": const("P1"),
                    "P3": lambda t: "away" if int(t) % 7 == 0 else "P1", "P4": const("screen")}, seconds=95)
    base = analyze(rows, ROOM, SocialConfig())
    for win in (5, 10, 30):
        summ = analyze(rows, ROOM, SocialConfig(window_s=win))
        assert summ["per_person"] == base["per_person"]          # whole-session scores don't depend on windows
        assert len(summ["windows"]) == math.ceil(95 / win)
        for w in summ["windows"]:
            for block in ("attention", "dominance", "engagement"):
                for v in w[block].values():
                    assert v == v and 0 <= v <= 1                  # no NaN, in range


def test_t27_group_size_warning():
    two = meeting({"P1": const("P2"), "P2": const("P1")}, seconds=5, people=("P1", "P2"))
    six = meeting({f"P{i}": const("P1") for i in range(1, 7)}, seconds=5, people=tuple(f"P{i}" for i in range(1, 7)))
    four = meeting({p: const("P1") for p in ("P1", "P2", "P3", "P4")}, seconds=5)
    assert "3-5 people" in analyze(two, None)["warnings"][0]
    assert "3-5 people" in analyze(six, None)["warnings"][0]
    assert analyze(four, None)["warnings"] == []


def test_t28_everyone_stares_at_p1():
    face_p1 = {p: (lambda p=p: lambda t: (0.0, direction_to(ROOM.seats[p], ROOM.seats["P1"])))()
               for p in ("P2", "P3", "P4")}
    rows = meeting({"P1": const("P2"), "P2": const("P1"), "P3": const("P1"), "P4": const("P1")}, torso=face_p1)
    summ = analyze(rows, ROOM, SocialConfig())
    assert summ["group"]["dominant"] == "P1"
    assert summ["per_person"]["P1"]["dominance"] == pytest.approx(1.0)
    assert summ["per_person"]["P1"]["attention_received"] == pytest.approx(1.0)


def test_turning_out_lowers_engagement():
    torso = {p: const((0.0, None)) for p in ("P1", "P2", "P3")}
    torso["P4"] = lambda t: ((90.0 if t >= 80 else 0.0), None)   # turned away for the last 40 s of 120
    rows = meeting({p: const("P1" if p != "P1" else "P2") for p in ("P1", "P2", "P3", "P4")}, torso=torso)
    eng = analyze(rows, ROOM, SocialConfig())["per_person"]
    assert eng["P4"]["engagement"] < eng["P3"]["engagement"]


def test_mutual_gaze_rate():
    s = Series.from_rows(meeting({"P1": const("P2"), "P2": const("P1"), "P3": const("P1"), "P4": const("P3")},
                                 seconds=4))
    assert mutual_gaze_rate(s) == pytest.approx(1 / 6)     # only the P1-P2 pair of 6 pairs


def test_seat_baseline_uses_other_meetings_only():
    from grouppulse.social.report import seat_baseline

    truths = {"A": ["P1", "P2", "P3", "P4"], "B": ["P1", "P3", "P2", "P4"], "C": ["P4", "P3", "P2", "P1"]}
    assert seat_baseline(truths, "C")[0] == "P1"     # P1 led A and B; C's own answer is not used
    assert seat_baseline(truths, "A")[0] in ("P3", "P1")
    assert sorted(seat_baseline(truths, "B")) == ["P1", "P2", "P3", "P4"]


def test_spearman():
    assert spearman(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1
    assert spearman(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == -1


def test_true_gaze_replaces_targets(ann_root):
    rows = [TrackRow("TS9999a", int(t * 25), t, "P1", True, 0.0, 0.0, None, "P4", None, None, 0.0, False)
            for t in (1.0, 5.0, 10.0, 20.0)]
    got = [r.gaze_target for r in with_true_gaze(rows, ann_root, "TS9999a")]
    assert got == ["table", "P2", "screen", "unknown"]
