"""A small IoU tracker that keeps person IDs stable through short occlusions.

Ultralytics' ByteTrack is used when available (backends.py); this is the dependency-free
fallback and the reference the tests exercise. Boxes are (x1, y1, x2, y2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    id: int
    box: Box
    misses: int = 0


@dataclass
class IoUTracker:
    min_iou: float = 0.3
    max_misses: int = 15          # frames a track survives without a match (0.6 s at 25 fps)
    _tracks: list[_Track] = field(default_factory=list)
    _next: int = 1

    def update(self, boxes: list[Box]) -> list[int]:
        """Assign a track id to each box (same order as `boxes`)."""
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self._tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        ids: list[int | None] = [None] * len(boxes)
        used_t: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.min_iou:
                break
            if ti in used_t or ids[bi] is not None:
                continue
            used_t.add(ti)
            t = self._tracks[ti]
            t.box, t.misses = boxes[bi], 0
            ids[bi] = t.id
        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        for bi, b in enumerate(boxes):
            if ids[bi] is None:
                self._tracks.append(_Track(self._next, b))
                ids[bi] = self._next
                self._next += 1
        return [int(i) for i in ids]
