"""Download AMI close-ups and run perception for many meetings, restartably.

    python scripts/build_tracks.py --data path/to/ami-data --meetings focus \
        --out path/to/ami-data/tracks_idiap --fps 2.5 --torso none

The next meeting downloads while the current one is processed. A meeting whose track table
already exists is skipped, and finished downloads are kept, so the script can simply be run
again after an interruption. Progress goes to <out>/build.log.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from grouppulse.data import ami_download, load_config  # noqa: E402
from grouppulse.data.meeting import video_path  # noqa: E402
from grouppulse.perception import run as perception_run  # noqa: E402

CLOSEUPS = ["Closeup1", "Closeup2", "Closeup3", "Closeup4"]


def log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (out / "build.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(data: Path, meeting: str) -> int:
    return ami_download.main(["--meetings", meeting, "--views", *CLOSEUPS, "--out", str(data)])


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--meetings", nargs="+", required=True, help="ids or set names from configs/ami.toml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", default="2.5")
    ap.add_argument("--torso", default="none", choices=["closeup", "room", "none"])
    ap.add_argument("--only-site", default="IS", help="meeting-id prefix to keep (default IS; '' for all)")
    args = ap.parse_args(argv)

    meetings = [m for m in cfg.meetings(args.meetings) if m.startswith(args.only_site)]
    todo = [m for m in meetings if not (args.out / f"{m}.csv").exists()]
    args.out.mkdir(parents=True, exist_ok=True)
    log(args.out, f"start: {len(meetings)} meetings, {len(todo)} to do: {' '.join(todo)}")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(fetch, args.data, todo[0]) if todo else None
        for k, m in enumerate(todo):
            pending.result()
            pending = pool.submit(fetch, args.data, todo[k + 1]) if k + 1 < len(todo) else None
            missing = [v for v in CLOSEUPS if not video_path(args.data, m, v).exists()]
            if missing:
                # never write a partial table: it would look finished and be skipped next time
                log(args.out, f"{m}: SKIPPED, missing {' '.join(missing)}; run again to retry")
                continue
            log(args.out, f"{m}: downloaded; processing")
            t0 = time.time()
            try:
                perception_run.main(["--data", str(args.data), "--meeting", m, "--fps", args.fps,
                                     "--torso", args.torso, "--gaze-model", "geometry",
                                     "--out", str(args.out / f"{m}.csv")])
                log(args.out, f"{m}: done in {time.time() - t0:.0f} s")
            except Exception:
                log(args.out, f"{m}: FAILED\n{traceback.format_exc()}")
    log(args.out, "finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
