"""Top-down seat layout of a meeting room, used to turn head pose into a gaze target.

AMI rooms are fixed: four seats around a table, one close-up camera per seat, a projector screen
and a whiteboard. Positions live in configs/seat_layouts.toml and are first estimates to be
calibrated against video (see notebooks/).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..data.config import REPO_ROOT
from .geometry import wrap180

DEFAULT_LAYOUTS = REPO_ROOT / "configs" / "seat_layouts.toml"


@dataclass(frozen=True)
class SeatLayout:
    name: str
    seats: dict[str, tuple[float, float]]          # person label -> (x, y) in metres
    objects: dict[str, tuple[float, float]]        # "screen", "whiteboard" -> (x, y)
    camera_of: dict[str, str]                      # person label -> close-up view name
    camera_dir: dict[str, float]                   # view name -> top-down direction camera looks (deg)

    def world_gaze(self, view: str, yaw: float) -> float:
        """Top-down gaze direction for a camera-relative head yaw seen from `view`.

        Facing the camera (yaw 0) = looking back along the camera axis; turning toward image
        right (+yaw) rotates toward the camera's right-hand side."""
        return wrap180(self.camera_dir[view] + 180.0 + yaw)


def load_layout(name: str, path: str | Path | None = None) -> SeatLayout:
    with Path(path or DEFAULT_LAYOUTS).open("rb") as f:
        raw = tomllib.load(f)[name]
    return SeatLayout(
        name=name,
        seats={k: tuple(v) for k, v in raw["seats"].items()},
        objects={k: tuple(v) for k, v in raw.get("objects", {}).items()},
        camera_of=dict(raw["camera_of"]),
        camera_dir={k: float(v) for k, v in raw["camera_dir"].items()},
    )


def layout_for_meeting(meeting: str, path: str | Path | None = None) -> SeatLayout:
    """ES and TS rooms share one layout guess; IS rooms another."""
    return load_layout("idiap" if meeting.startswith("IS") else "edinburgh", path)
