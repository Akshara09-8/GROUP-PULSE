"""T0.1 (loader) and T0.4: a meeting loads; a missing camera view warns and is skipped."""

import logging

from grouppulse.data import load_meeting
from grouppulse.data.meeting import video_path


def test_missing_views_warn_not_crash(tmp_path, ann_root, caplog):
    p = video_path(tmp_path, "TS9999a", "Closeup1")
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not really a video")
    with caplog.at_level(logging.WARNING):
        m = load_meeting(tmp_path, ann_root, "TS9999a", ["Closeup1", "Closeup2", "Corner"])
    assert list(m.videos) == ["Closeup1"]
    assert m.missing_views == ["Closeup2", "Corner"]
    assert "Corner" in caplog.text
    assert len(m.segments) == 5
    assert m.speakers["A"]["role"] == "PM"


def test_meeting_without_speaker_table_still_loads(tmp_path, ann_root, caplog):
    import shutil

    copy = tmp_path / "ann"
    shutil.copytree(ann_root / "segments", copy / "segments")
    with caplog.at_level(logging.WARNING):
        m = load_meeting(tmp_path, copy, "TS9999a", ["Corner"])
    assert m.speakers == {}
    assert m.last_annotation == 29.3
