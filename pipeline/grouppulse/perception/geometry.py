"""Angle helpers. All angles are degrees.

Conventions used across Group Pulse:
- Camera-relative yaw: 0 = facing the camera, +90 = facing image right, 180 = back to camera.
- Pitch: + = looking up, - = looking down.
- Top-down (world) angles: counter-clockwise from +x, with +y pointing away from the screen end of the room.
"""

from __future__ import annotations

import math

BIN8_NAMES = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")


def wrap180(a: float) -> float:
    """Map any angle to (-180, 180]."""
    a = math.fmod(a + 180.0, 360.0)
    if a <= 0:
        a += 360.0
    return a - 180.0


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (0..180)."""
    return abs(wrap180(a - b))


def bin8(yaw: float) -> int:
    """8-way direction bin for a camera-relative yaw: 0 = front, 2 = right, 4 = back, 6 = left."""
    return int(((yaw % 360.0) + 22.5) // 45.0) % 8


def bin8_name(yaw: float) -> str:
    return BIN8_NAMES[bin8(yaw)]


def direction_to(origin: tuple[float, float], target: tuple[float, float]) -> float:
    """Top-down angle from origin to target."""
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))
