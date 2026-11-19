"""Gaze target: who or what a person is looking at, from a top-down gaze direction and pitch.

A cone test: the target is the person or object whose direction from the viewer is closest to
the gaze direction, if it is within the cone. Looking clearly down = "table" (notes, laptop,
phone). Nothing in the cone = "away". No face = "unknown" (test T1.8).
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import angle_diff, direction_to
from .layout import SeatLayout

TABLE, AWAY, UNKNOWN = "table", "away", "unknown"


@dataclass(frozen=True)
class GazeResult:
    target: str
    off_angle: float | None   # degrees between gaze and the chosen target (None for table/away/unknown)


def gaze_target(layout: SeatLayout, viewer: str, gaze_dir: float | None, pitch: float | None,
                cone_deg: float = 25.0, table_pitch: float = -20.0) -> GazeResult:
    if gaze_dir is None or pitch is None:
        return GazeResult(UNKNOWN, None)
    if pitch <= table_pitch:
        return GazeResult(TABLE, None)
    origin = layout.seats[viewer]
    candidates = {k: v for k, v in layout.seats.items() if k != viewer} | layout.objects
    best, best_off = AWAY, None
    for name, pos in candidates.items():
        off = angle_diff(gaze_dir, direction_to(origin, pos))
        if off <= cone_deg and (best_off is None or off < best_off):
            best, best_off = name, off
    return GazeResult(best, best_off)
