"""Read basic video facts (frame count, fps, duration) with OpenCV or ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frames: int
    fps: float

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


class VideoProbeError(RuntimeError):
    pass


def _probe_cv2(path: Path) -> VideoInfo | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoProbeError(f"OpenCV could not open {path}")
    try:
        return VideoInfo(path, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS)))
    finally:
        cap.release()


def _probe_ffprobe(path: Path) -> VideoInfo | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise VideoProbeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    stream = json.loads(out.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return VideoInfo(path, int(stream["nb_read_packets"]), float(num) / float(den))


def probe(path: str | Path) -> VideoInfo:
    """Frame count and fps for a video. Needs `pip install .[video]` or ffprobe on PATH."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    info = _probe_cv2(p) or _probe_ffprobe(p)
    if info is None:
        raise VideoProbeError("Install OpenCV (pip install -e .[video]) or ffmpeg to read videos")
    return info


def frames_match_duration(info: VideoInfo, expected_duration: float, tolerance_frames: int = 2) -> bool:
    """T0.1: does the frame count agree with the expected duration (within a couple of frames)?"""
    return abs(info.frames - expected_duration * info.fps) <= tolerance_frames
