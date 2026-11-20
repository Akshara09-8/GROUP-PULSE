"""python -m grouppulse.evaluation --data D --idiap-tracks D/tracks_idiap --edinburgh-tracks D/tracks_full --out reports [--repro]"""

import argparse
import hashlib
from pathlib import Path

from .report import build
from .run import dumps, run

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--idiap-tracks", type=Path)
ap.add_argument("--edinburgh-tracks", type=Path)
ap.add_argument("--out", type=Path, default=Path("reports"))
ap.add_argument("--repro", action="store_true", help="run everything twice and compare (T5.4)")
args = ap.parse_args()

first = dumps(run(args.data, args.idiap_tracks, args.edinburgh_tracks))
repro = None
if args.repro:
    repro = first == dumps(run(args.data, args.idiap_tracks, args.edinburgh_tracks))
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "phase5_results.json").write_text(first, encoding="utf-8")
import json  # noqa: E402
(args.out / "phase5_report.md").write_text(build(json.loads(first), repro), encoding="utf-8")
print(f"results sha256 {hashlib.sha256(first.encode()).hexdigest()[:16]}"
      + ("" if repro is None else f"; reproducible: {repro}"))
print(f"wrote {args.out / 'phase5_report.md'} and {args.out / 'phase5_results.json'}")
