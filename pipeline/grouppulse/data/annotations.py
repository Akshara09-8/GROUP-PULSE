"""Parse AMI NXT annotations: speaker segments and the speaker table.

Expected layout after unzipping ami_public_manual_1.6.2.zip:
    <ann_root>/segments/ES2002a.A.segments.xml   (one file per speaker letter A-E)
    <ann_root>/corpusResources/meetings.xml
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SPEAKER_LETTERS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True, order=True)
class Segment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _local(tag: str) -> str:
    """Strip an XML namespace: '{ns}segment' -> 'segment'."""
    return tag.rsplit("}", 1)[-1]


def _attr(el: ET.Element, name: str) -> str | None:
    """Read an attribute whether or not it carries a namespace prefix."""
    for key, value in el.attrib.items():
        if _local(key) == name:
            return value
    return None


def find_ann_root(path: str | Path) -> Path:
    """Return the folder that holds `segments/`, searching below `path` (zips may add a top folder)."""
    p = Path(path)
    if (p / "segments").is_dir():
        return p
    for seg in sorted(p.rglob("segments")):
        if seg.is_dir():
            return seg.parent
    raise FileNotFoundError(f"No 'segments' folder under {p}; did the annotations download finish?")


def load_segments(ann_root: str | Path, meeting: str) -> list[Segment]:
    """All speaker segments for one meeting, sorted by start time.

    Segments with missing or reversed times are skipped (AMI has a few)."""
    seg_dir = Path(ann_root) / "segments"
    out: list[Segment] = []
    for letter in SPEAKER_LETTERS:
        path = seg_dir / f"{meeting}.{letter}.segments.xml"
        if not path.exists():
            continue
        for el in ET.parse(path).getroot().iter():
            if _local(el.tag) != "segment":
                continue
            start, end = _attr(el, "transcriber_start"), _attr(el, "transcriber_end")
            if start is None or end is None:
                continue
            s, e = float(start), float(end)
            if e > s:
                out.append(Segment(s, e, letter))
    if not out:
        raise FileNotFoundError(f"No segment files for {meeting} in {seg_dir}")
    return sorted(out)


def speaking_time(segments: list[Segment]) -> dict[str, float]:
    """Total seconds spoken per speaker letter (overlaps inside one speaker are merged)."""
    by_spk: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        by_spk[s.speaker].append(s)
    totals: dict[str, float] = {}
    for spk, segs in by_spk.items():
        total, cur_s, cur_e = 0.0, None, None
        for s in sorted(segs):
            if cur_e is None or s.start > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = s.start, s.end
            else:
                cur_e = max(cur_e, s.end)
        total += cur_e - cur_s
        totals[spk] = round(total, 3)
    return dict(sorted(totals.items()))


@dataclass(frozen=True, order=True)
class FocusSpan:
    start: float
    end: float
    speaker: str      # who is looking (letter A-E)
    target: str       # a speaker letter, or "table", "screen", "whiteboard", "away"


_PLACES = {"slide-screen": "screen", "table": "table", "whiteboard": "whiteboard"}


def load_focus(ann_root: str | Path, meeting: str) -> list[FocusSpan]:
    """AMI focus-of-attention labels (focus/<meeting>.<letter>.foa.xml), only for 14 meetings.

    'unspecified' (looking at nothing in particular) maps to "away"."""
    out: list[FocusSpan] = []
    for letter in SPEAKER_LETTERS:
        path = Path(ann_root) / "focus" / f"{meeting}.{letter}.foa.xml"
        if not path.exists():
            continue
        for el in ET.parse(path).getroot().iter():
            if _local(el.tag) != "foa":
                continue
            s, e = _attr(el, "starttime"), _attr(el, "endtime")
            if s is None or e is None or float(e) <= float(s):
                continue
            kind = _attr(el, "type")
            if kind == "person":
                target = _attr(el, "nxt_agent") or "away"
            elif kind == "place":
                target = _PLACES.get(_attr(el, "place") or "", "away")
            else:
                target = "away"
            out.append(FocusSpan(float(s), float(e), letter, target))
    if not out:
        raise FileNotFoundError(f"No focus-of-attention labels for {meeting}")
    return sorted(out)


def seat_of_speaker(speakers: dict[str, dict[str, str]]) -> dict[str, str]:
    """Speaker letter -> seat label P1-P4, using the close-up camera that films them.

    AMI letters are not in camera order (ES2002a: B is filmed by Closeup4), so Group Pulse
    names people by seat (Closeup<N> -> P<N>)."""
    out: dict[str, str] = {}
    for letter, attrs in speakers.items():
        cam = attrs.get("camera", "")
        if cam.startswith("Closeup") and cam[7:].isdigit():
            out[letter] = f"P{cam[7:]}"
    return out


def load_speakers(ann_root: str | Path, meeting: str) -> dict[str, dict[str, str]]:
    """Speaker table for one meeting from corpusResources/meetings.xml.

    Returns {letter: {attribute: value}}, e.g. {"A": {"global_name": "FEE005", "camera": "Closeup1", ...}}.
    Attribute names are passed through as found, so new fields need no code change."""
    path = Path(ann_root) / "corpusResources" / "meetings.xml"
    root = ET.parse(path).getroot()
    for m in root.iter():
        if _local(m.tag) != "meeting" or _attr(m, "observation") != meeting:
            continue
        table: dict[str, dict[str, str]] = {}
        for sp in m:
            if _local(sp.tag) != "speaker":
                continue
            attrs = {_local(k): v for k, v in sp.attrib.items()}
            letter = attrs.get("nxt_agent")
            if letter:
                table[letter] = attrs
        return table
    raise KeyError(f"{meeting} not found in {path}")
