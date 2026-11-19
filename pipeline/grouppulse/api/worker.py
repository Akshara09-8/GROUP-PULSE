"""Background processing: queued -> processing -> done | failed (test T3.2).

One worker thread, so the free server never runs two heavy jobs at once. A job's processor turns
(video path, people count) into an analyze() summary. Jobs live in memory: if the server restarts,
queued jobs are marked failed at startup so nothing stays "processing" forever.
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .store import Store

log = logging.getLogger(__name__)

Processor = Callable[[Path, int], dict]


class ProcessingUnavailable(RuntimeError):
    """The server cannot process this kind of upload (yet); shown to the user as-is."""


def single_camera_processor(video: Path, people: int) -> dict:
    """Placeholder until the single-camera perception path exists.

    Phase 1 reads one close-up camera per person (the AMI setup). A demo upload is one camera
    filming the whole group, which needs multi-face tracking and image-plane gaze targets."""
    raise ProcessingUnavailable(
        "Uploads from a single camera are not supported yet. Try one of the AMI demo sessions.")


class Worker:
    def __init__(self, store: Store, processor: Processor):
        self.store, self.processor = store, processor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gp-worker")

    def submit(self, sid: str, video: str, people: int) -> Future:
        return self._pool.submit(self._run, sid, Path(video), people)

    def _run(self, sid: str, video: Path, people: int) -> None:
        self.store.set_status(sid, "processing")
        try:
            summary = self.processor(video, people)
            self.store.save_summary(sid, summary)
        except ProcessingUnavailable as e:
            self.store.set_status(sid, "failed", str(e))
        except Exception as e:  # never leave a job stuck in "processing"
            log.error("session %s failed:\n%s", sid, traceback.format_exc())
            self.store.set_status(sid, "failed", f"Processing failed: {type(e).__name__}")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def fail_orphans(store: Store) -> int:
    """Mark sessions left queued/processing by a previous run as failed."""
    n = 0
    for s in store.list_sessions():
        if s["status"] in ("queued", "processing"):
            store.set_status(s["id"], "failed", "The server restarted before this finished. Please upload again.")
            n += 1
    return n
