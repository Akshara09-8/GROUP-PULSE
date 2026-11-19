"""T0.2: annotation times line up with video frames within 40 ms."""

from grouppulse.data import Segment, frame_to_time, max_rounding_error, segments_past_end, time_to_frame
from grouppulse.data.align import MAX_DRIFT_S


def test_round_trip_within_half_a_frame(ann_root):
    from grouppulse.data import load_segments

    segs = load_segments(ann_root, "TS9999a")
    err = max_rounding_error(segs, fps=25.0)
    assert err <= 0.5 / 25 + 1e-9  # 20 ms at 25 fps
    assert err <= MAX_DRIFT_S


def test_frame_mapping():
    assert time_to_frame(1.0, 25) == 25
    assert time_to_frame(1.019, 25) == 25
    assert time_to_frame(1.021, 25) == 26
    assert frame_to_time(50, 25) == 2.0


def test_segments_past_end_flags_drift():
    segs = [Segment(0, 10, "A"), Segment(50, 60.03, "B"), Segment(55, 60.2, "C")]
    late = segments_past_end(segs, video_duration=60.0)
    assert late == [Segment(55, 60.2, "C")]  # 30 ms over is inside tolerance, 200 ms is not


def test_cut_off_last_utterance_is_not_drift():
    from grouppulse.data.align import is_truncated_ending

    # ES2003a: the final utterance starts at 1118.6 s and the recording stops at 1125.8 s
    assert is_truncated_ending([Segment(1118.56, 1126.73, "B")], 1125.8)
    assert not is_truncated_ending([Segment(1130.0, 1131.0, "B")], 1125.8)   # starts after the end
    assert not is_truncated_ending([Segment(1100.0, 1130.0, "B")], 1125.8)   # 4 s over
    assert not is_truncated_ending([], 1125.8)
