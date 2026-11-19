"""AMI focus-of-attention labels as T1.5 ground truth, mapped from speaker letters to seats."""

from grouppulse.data import load_focus, load_speakers, seat_of_speaker
from grouppulse.perception.evaluate import paired, truth_at
from grouppulse.perception.io import TrackRow


def row(t, person, target):
    return TrackRow("TS9999a", int(t * 25), t, person, True, 0.0, 0.0, 0.0, target, 0.0, None, 0.0, False)


def test_load_focus(ann_root):
    spans = load_focus(ann_root, "TS9999a")
    assert [s.target for s in spans] == ["table", "B", "screen", "away"]
    assert truth_at(spans, 5.0).target == "B"
    assert truth_at(spans, 20.0) is None


def test_seats_follow_cameras_not_letters(ann_root):
    seat = seat_of_speaker(load_speakers(ann_root, "TS9999a"))
    assert seat == {"A": "P1", "B": "P2"}
    real_style = {"B": {"camera": "Closeup4"}, "A": {"camera": "Closeup1"}}
    assert seat_of_speaker(real_style) == {"B": "P4", "A": "P1"}


def test_pairing_converts_letters_to_seats(ann_root):
    seat = seat_of_speaker(load_speakers(ann_root, "TS9999a"))
    rows = [row(1.0, "P1", "table"), row(5.0, "P1", "P2"), row(10.0, "P1", "P3"), row(30.0, "P1", "away")]
    true, pred = paired(rows, load_focus(ann_root, "TS9999a"), seat)
    assert true == ["table", "P2", "screen"]
    assert pred == ["table", "P2", "P3"]
