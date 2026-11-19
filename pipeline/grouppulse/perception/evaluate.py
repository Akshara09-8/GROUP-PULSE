"""Score a track table against AMI focus-of-attention labels (test T1.5).

    python -m grouppulse.perception.evaluate --data path/to/ami-data --meeting ES2008a

Only the 14 meetings in the `focus` set of configs/ami.toml have these labels.
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

from ..data.annotations import FocusSpan, find_ann_root, load_focus, load_speakers, seat_of_speaker
from ..data.config import REPO_ROOT
from .io import TrackRow, read_tracks
from .metrics import confusion, target_accuracy

T15_TARGET = 0.70


def truth_at(spans: list[FocusSpan], t: float) -> FocusSpan | None:
    i = bisect.bisect_right([s.start for s in spans], t) - 1
    return spans[i] if i >= 0 and spans[i].start <= t < spans[i].end else None


def paired(rows: list[TrackRow], focus: list[FocusSpan], seat: dict[str, str]) -> tuple[list[str], list[str]]:
    """Ground truth and predicted targets for every row that has a label, both in seat names."""
    by_seat: dict[str, list[FocusSpan]] = {}
    for f in focus:
        if f.speaker in seat:
            by_seat.setdefault(seat[f.speaker], []).append(f)
    true, pred = [], []
    for r in rows:
        span = truth_at(by_seat.get(r.person, []), r.t)
        if span is None:
            continue
        true.append(seat.get(span.target, span.target))
        pred.append(r.gaze_target)
    return true, pred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--tracks", type=Path, default=None)
    args = ap.parse_args(argv)

    ann = find_ann_root(args.data / "annotations")
    seat = seat_of_speaker(load_speakers(ann, args.meeting))
    rows = read_tracks(args.tracks or REPO_ROOT / "data" / "tracks" / f"{args.meeting}.csv")
    true, pred = paired(rows, load_focus(ann, args.meeting), seat)
    acc = target_accuracy(true, pred)
    print(f"{len(true)} labelled frames; gaze-target accuracy {acc:.1%} (T1.5 needs {T15_TARGET:.0%})")
    for t, row in sorted(confusion(true, pred).items()):
        print(f"  {t:>10}: " + ", ".join(f"{k} {v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))
    return 0 if acc >= T15_TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
