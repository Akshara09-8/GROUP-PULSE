"""Map annotation times (seconds) to video frames and check they line up.

Test T0.2 asks for drift <= 40 ms. Two things can break alignment:
1. rounding a time to the nearest frame (at most half a frame: 20 ms at 25 fps), and
2. annotations that run past the end of the video (a sign of an offset or wrong file).
"""

from __future__ import annotations

from typing import Iterable

from .annotations import Segment

MAX_DRIFT_S = 0.040


def time_to_frame(t: float, fps: float) -> int:
    return int(round(t * fps))


def frame_to_time(frame: int, fps: float) -> float:
    return frame / fps


def max_rounding_error(segments: Iterable[Segment], fps: float) -> float:
    """Largest gap (seconds) between a segment boundary and the frame it maps to."""
    worst = 0.0
    for s in segments:
        for t in (s.start, s.end):
            worst = max(worst, abs(t - frame_to_time(time_to_frame(t, fps), fps)))
    return worst


def segments_past_end(segments: Iterable[Segment], video_duration: float,
                      tolerance: float = MAX_DRIFT_S) -> list[Segment]:
    """Segments that end after the video does (beyond the tolerance)."""
    return [s for s in segments if s.end > video_duration + tolerance]


def is_truncated_ending(late: list[Segment], video_duration: float, max_overrun: float = 2.0) -> bool:
    """True when the only overruns are utterances still going when the recording stopped:
    each starts before the video ends and ends at most `max_overrun` seconds after it.
    That is a cut-off ending, not a timing offset, so it should warn rather than fail T0.2."""
    return bool(late) and all(s.start < video_duration and s.end - video_duration <= max_overrun for s in late)
