"""Download AMI annotations and videos.

    python -m grouppulse.data.ami_download --dry-run
    python -m grouppulse.data.ami_download --annotations --meetings demo --views Closeup1 Corner

Files already on disk with the right size are skipped, so the command can be re-run after a break.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import REPO_ROOT, load_config
from .meeting import video_path

DEFAULT_OUT = REPO_ROOT / "data" / "raw"
CHUNK = 1 << 20


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f"  have  {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        done = 0
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            if size:
                print(f"\r  get   {dest.name}  {done * 100 // size:3d}%", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


RETRY_WAITS_S = (60, 300, 900)   # the AMI server starts refusing (HTTP 403) after a few GB in a row
PAUSE_S = 5                       # short gap between files to stay under that limit


def download_with_retry(url: str, dest: Path, waits=RETRY_WAITS_S, pause: float = PAUSE_S) -> Path:
    """download() with back-off on HTTP 403/429/5xx and timeouts; other errors are raised at once."""
    import time
    import urllib.error

    for attempt, wait in enumerate((*waits, None)):
        try:
            got = download(url, dest)
            time.sleep(pause)
            return got
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError, OSError) as e:
            code = getattr(e, "code", None)
            retryable = code in (403, 429) or (code is not None and code >= 500) or code is None
            if not retryable or wait is None:
                raise
            print(f"  retry {Path(url).name} in {wait} s ({e})", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def plan(meetings: list[str], views: list[str], annotations: bool, out: Path, cfg) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    if annotations:
        jobs.append((cfg.annotations_url, out / "annotations" / Path(cfg.annotations_url).name))
    for m in meetings:
        for v in cfg.views_for(m, views):
            jobs.append((cfg.video_url_for(m, v), video_path(out, m, v)))
    return jobs


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", nargs="+", default=["demo"], help="meeting ids or set names from configs/ami.toml")
    ap.add_argument("--views", nargs="+", default=list(cfg.views),
                    help='camera views; "room" = the meeting site\'s room camera (Corner for ES, C for IS)')
    ap.add_argument("--annotations", action="store_true", help="also fetch and unzip the manual annotations")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded and stop")
    args = ap.parse_args(argv)

    jobs = plan(cfg.meetings(args.meetings), args.views, args.annotations or args.dry_run, args.out, cfg)
    if args.dry_run:
        for url, dest in jobs:
            print(f"{url}\n  -> {dest}")
        print(f"{len(jobs)} files")
        return 0

    for url, dest in jobs:
        try:
            download_with_retry(url, dest)
        except Exception as e:  # keep going: one missing view should not stop the rest
            print(f"  FAILED {url}: {e}", file=sys.stderr)
    if args.annotations:
        z = args.out / "annotations" / Path(cfg.annotations_url).name
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(args.out / "annotations")
            print(f"  unzipped annotations into {args.out / 'annotations'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
