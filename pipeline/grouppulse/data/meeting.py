"""Load one AMI meeting: which camera views exist, speaker segments and speaker table."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .annotations import Segment, load_segments, load_speakers, speaking_time

log = logging.getLogger(__name__)


@dataclass
class Meeting:
    id: str
    videos: dict[str, Path]
    segments: list[Segment]
    speakers: dict[str, dict[str, str]]
    missing_views: list[str] = field(default_factory=list)

    @property
    def speaking_time(self) -> dict[str, float]:
        return speaking_time(self.segments)

    @property
    def last_annotation(self) -> float:
        return max(s.end for s in self.segments)


def video_path(raw_root: str | Path, meeting: str, view: str) -> Path:
    """Where ami_download puts a video: <raw>/amicorpus/<meeting>/video/<meeting>.<view>.avi"""
    return Path(raw_root) / "amicorpus" / meeting / "video" / f"{meeting}.{view}.avi"


def load_meeting(raw_root: str | Path, ann_root: str | Path, meeting: str,
                 views: list[str] | tuple[str, ...]) -> Meeting:
    """Load a meeting. A missing camera view is logged and skipped, never fatal (test T0.4)."""
    videos: dict[str, Path] = {}
    missing: list[str] = []
    for view in views:
        p = video_path(raw_root, meeting, view)
        if p.exists():
            videos[view] = p
        else:
            missing.append(view)
            log.warning("%s: camera view %s not found at %s; skipping it", meeting, view, p)
    try:
        speakers = load_speakers(ann_root, meeting)
    except (FileNotFoundError, KeyError) as e:
        log.warning("%s: no speaker table (%s)", meeting, e)
        speakers = {}
    return Meeting(meeting, videos, load_segments(ann_root, meeting), speakers, missing)
