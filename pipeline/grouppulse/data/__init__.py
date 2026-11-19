"""AMI Meeting Corpus loading: config, download, annotations, video probing, alignment and splits."""

from .config import AmiConfig, load_config
from .annotations import (FocusSpan, Segment, find_ann_root, load_focus, load_segments,
                          load_speakers, seat_of_speaker, speaking_time)
from .align import frame_to_time, max_rounding_error, segments_past_end, time_to_frame
from .meeting import Meeting, load_meeting
from .splits import series_of, split_meetings

__all__ = [
    "AmiConfig", "load_config",
    "FocusSpan", "Segment", "find_ann_root", "load_focus", "load_segments", "load_speakers",
    "seat_of_speaker", "speaking_time",
    "frame_to_time", "max_rounding_error", "segments_past_end", "time_to_frame",
    "Meeting", "load_meeting",
    "series_of", "split_meetings",
]
