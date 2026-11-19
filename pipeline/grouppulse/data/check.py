"""Phase 0 data check (tests T0.1, T0.2, T0.4 on real downloaded data).

    python -m grouppulse.data.check --data data/raw --meetings demo
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .annotations import find_ann_root
from .align import MAX_DRIFT_S, is_truncated_ending, max_rounding_error, segments_past_end
from .config import REPO_ROOT, load_config
from .meeting import load_meeting
from .video import VideoProbeError, probe


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meetings", nargs="+", default=["demo"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="  warn  %(message)s")

    try:
        ann_root = find_ann_root(args.data / "annotations")
    except FileNotFoundError as e:
        print(f"FAIL  {e}")
        return 1
    failures = 0
    for mid in cfg.meetings(args.meetings):
        print(f"{mid}")
        try:
            m = load_meeting(args.data, ann_root, mid, cfg.views_for(mid))
        except FileNotFoundError as e:
            print(f"  FAIL  annotations: {e}")
            failures += 1
            continue
        print(f"  ok    {len(m.segments)} segments, speakers {sorted(m.speaking_time)}, last at {m.last_annotation:.1f} s")
        err = max_rounding_error(m.segments, cfg.fps)
        failures += err > MAX_DRIFT_S
        print(f"  {'ok  ' if err <= MAX_DRIFT_S else 'FAIL'}  frame rounding error {err * 1000:.1f} ms (limit 40)")
        for view, path in m.videos.items():
            try:
                info = probe(path)
            except VideoProbeError as e:
                print(f"  skip  {view}: {e}")
                continue
            late = segments_past_end(m.segments, info.duration)
            cut = is_truncated_ending(late, info.duration)
            state = "ok  " if not late else "warn" if cut else "FAIL"
            failures += bool(late) and not cut
            note = (f"recording stops {max(s.end for s in late) - info.duration:.1f} s before the last utterance ends"
                    if cut else f"{len(late)} segments past the end")
            print(f"  {state}  {view}: {info.frames} frames at {info.fps:g} fps = {info.duration:.1f} s; {note}")
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
