"""T1.1 / T1.2 logic: detection recall, stable IDs through occlusion; track table round trip."""

import numpy as np

from grouppulse.perception import IoUTracker, iou
from grouppulse.perception.io import TrackRow, read_tracks, write_tracks
from grouppulse.perception.labels import sample_frames
from grouppulse.perception.metrics import (confusion, detection_recall, id_switches,
                                           mean_angle_error, target_accuracy)


def box(cx, cy, w=80, h=160):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_iou():
    assert iou(box(0, 0), box(0, 0)) == 1.0
    assert iou(box(0, 0), box(500, 0)) == 0.0


def test_four_seated_people_keep_ids_through_occlusion():
    rng = np.random.default_rng(3)
    seats = [120, 260, 400, 540]
    tr = IoUTracker()
    rows = []
    for f in range(750):  # 30 s at 25 fps
        visible = [k for k in range(4) if not (k == 2 and 300 <= f < 310)]  # P3 hidden 10 frames
        boxes = [box(seats[k] + rng.normal(0, 4), 300 + rng.normal(0, 3)) for k in visible]
        ids = tr.update(boxes)
        rows += [(f, k, i) for k, i in zip(visible, ids)]
    assert id_switches(rows) == 0


def test_long_occlusion_is_one_switch_at_most():
    tr = IoUTracker(max_misses=5)
    rows = []
    for f in range(100):
        boxes = [box(100, 300)] + ([] if 40 <= f < 60 else [box(400, 300)])
        ids = tr.update(boxes)
        rows += [(f, 0, ids[0])] + ([(f, 1, ids[1])] if len(ids) > 1 else [])
    assert id_switches(rows) <= 1


def test_detection_recall():
    gt = {0: [box(100, 300), box(300, 300)], 1: [box(100, 300)]}
    pred = {0: [box(102, 301)], 1: [box(100, 300), box(600, 300)]}
    assert detection_recall(gt, pred) == 2 / 3


def test_other_metrics():
    assert mean_angle_error([10, 170], [20, -170]) == 15
    assert target_accuracy(["P1", "P2", "away"], ["P1", "P3", "away"]) == 2 / 3
    assert confusion(["P1", "P1"], ["P1", "P2"]) == {"P1": {"P1": 1, "P2": 1}}


def test_track_table_round_trip(tmp_path):
    rows = [
        TrackRow("ES2002a", 0, 0.0, "P1", True, 12.5, -3.0, 102.5, "P3", 12.5, 5.0, 0.9, False),
        TrackRow("ES2002a", 5, 0.2, "P2", False, None, None, None, "unknown", None, None, 0.0, True),
    ]
    p = write_tracks(rows, tmp_path / "t.csv")
    assert read_tracks(p) == rows


def test_label_sampling_spreads_frames():
    f = sample_frames(25 * 60 * 20, 10)
    assert len(f) == 10 and f[0] > 0 and f[-1] < 30000
    assert sample_frames(0, 10) == []
