"""Uploaded-video storage. Local folder now; Supabase Storage would implement the same 3 methods."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

CHUNK = 1 << 20

# First bytes of the container formats we accept (checked, not trusted from the file name).
_MAGIC = (
    (4, b"ftyp"),             # MP4 / MOV / M4V
    (0, b"RIFF"),             # AVI (RIFF....AVI )
    (0, b"\x1aE\xdf\xa3"),    # WebM / MKV
)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


class TooLarge(Exception):
    pass


class NotVideo(Exception):
    pass


def looks_like_video(head: bytes) -> bool:
    return any(head[off:off + len(sig)] == sig for off, sig in _MAGIC)


class LocalFiles:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, src: BinaryIO, name: str, suffix: str, max_bytes: int) -> tuple[str, int]:
        """Stream an upload to disk; reject non-video content and files over max_bytes."""
        suffix = suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise NotVideo(f"'{suffix or 'no extension'}' is not a supported video type")
        dest = self.root / f"{name}{suffix}"
        size = 0
        try:
            with dest.open("wb") as out:
                first = True
                while chunk := src.read(CHUNK):
                    if first and not looks_like_video(chunk[:16]):
                        raise NotVideo("the file does not look like a video")
                    first = False
                    size += len(chunk)
                    if size > max_bytes:
                        raise TooLarge(f"the file is over {max_bytes // (1 << 20)} MB")
                    out.write(chunk)
            if size == 0:
                raise NotVideo("the file is empty")
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        return str(dest), size

    def delete(self, path: str | None) -> None:
        """Delete a video and its overlay (per-frame face boxes are derived from it)."""
        if path:
            Path(path).unlink(missing_ok=True)
            Path(str(path) + ".overlay.json").unlink(missing_ok=True)

    def exists(self, path: str | None) -> bool:
        return bool(path) and Path(path).exists()
