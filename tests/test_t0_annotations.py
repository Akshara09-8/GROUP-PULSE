"""T0.1 (annotations part): AMI-style annotation files parse correctly."""

import pytest

from grouppulse.data import find_ann_root, load_segments, load_speakers, speaking_time


def test_segments_parse_and_sort(ann_root):
    segs = load_segments(ann_root, "TS9999a")
    # 3 good A segments + 2 good B segments; the A segment with no end and the zero-length B one are dropped
    assert len(segs) == 5
    assert [s.start for s in segs] == sorted(s.start for s in segs)
    assert {s.speaker for s in segs} == {"A", "B"}


def test_speaking_time_merges_overlaps(ann_root):
    totals = speaking_time(load_segments(ann_root, "TS9999a"))
    # A: 0.52-6.00 (two overlapping segments merged) + 20.00-25.517
    assert totals["A"] == pytest.approx(5.48 + 5.517)
    assert totals["B"] == pytest.approx(13.64 + 2.19)


def test_speaker_table(ann_root):
    table = load_speakers(ann_root, "TS9999a")
    assert table["A"]["global_name"] == "FEE001"
    assert table["B"]["camera"] == "Closeup2"


def test_unknown_meeting_raises(ann_root):
    with pytest.raises(FileNotFoundError):
        load_segments(ann_root, "XX0000z")
    with pytest.raises(KeyError):
        load_speakers(ann_root, "XX0000z")


def test_find_ann_root_handles_extra_top_folder(tmp_path, ann_root):
    nested = tmp_path / "annotations" / "ami_public_manual_1.6.2"
    (nested / "segments").mkdir(parents=True)
    assert find_ann_root(tmp_path / "annotations") == nested
    assert find_ann_root(ann_root) == ann_root
