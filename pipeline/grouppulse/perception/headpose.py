"""Head pose (yaw, pitch) from 2D facial landmarks with OpenCV solvePnP.

Works with the 5 landmarks YuNet returns (two eyes, nose tip, two mouth corners) or the 6-point
set (adds the chin) that MediaPipe/OpenFace landmarks can supply. The 3D model is a generic adult
face in millimetres, written in camera axes (x right, y down, z away from the camera), facing the
camera when rotation is identity. Points are ordered by *image side*, not by the person's side.
"""

from __future__ import annotations

import math

import numpy as np

# Generic face model (mm). Order: eye on image-left, eye on image-right, nose tip,
# mouth corner on image-left, mouth corner on image-right, chin.
MODEL_6 = np.array([
    [-165.0, -170.0, 135.0],
    [165.0, -170.0, 135.0],
    [0.0, 0.0, 0.0],
    [-150.0, 150.0, 125.0],
    [150.0, 150.0, 125.0],
    [0.0, 330.0, 65.0],
], dtype=np.float64)
MODEL_5 = MODEL_6[:5]


def camera_matrix(width: int, height: int, focal: float | None = None) -> np.ndarray:
    """Pinhole intrinsics. Without calibration, focal length ~ image width is a fair guess."""
    f = float(focal or width)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def _rot(yaw: float, pitch: float) -> np.ndarray:
    """Rotation for our yaw/pitch convention (see geometry.py): R = Ry(-yaw) @ Rx(-pitch)."""
    y, p = math.radians(-yaw), math.radians(-pitch)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def _euler(r: np.ndarray) -> tuple[float, float]:
    """Inverse of _rot (roll ignored): returns (yaw, pitch) in degrees."""
    p = math.atan2(-r[1, 2], r[1, 1])
    y = math.atan2(-r[2, 0], r[0, 0])
    return -math.degrees(y), -math.degrees(p)


def project_model(yaw: float, pitch: float, width: int = 640, height: int = 480,
                  distance_mm: float = 1500.0, points: int = 6) -> np.ndarray:
    """Synthetic landmarks for a head at (yaw, pitch); used by tests and for sanity checks."""
    import cv2

    model = MODEL_6 if points == 6 else MODEL_5
    rvec, _ = cv2.Rodrigues(_rot(yaw, pitch))
    tvec = np.array([[0.0], [0.0], [distance_mm]])
    pts, _ = cv2.projectPoints(model, rvec, tvec, camera_matrix(width, height), None)
    return pts.reshape(-1, 2)


def head_pose(landmarks: np.ndarray, width: int, height: int,
              focal: float | None = None) -> tuple[float, float] | None:
    """(yaw, pitch) in degrees from 5 or 6 image landmarks, or None if PnP fails."""
    import cv2

    pts = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    if len(pts) not in (5, 6) or not np.all(np.isfinite(pts)):
        return None
    model = MODEL_6 if len(pts) == 6 else MODEL_5
    k = camera_matrix(width, height, focal)
    # Start from "facing the camera at ~1.5 m"; keeps the 5-point solve from flipping.
    rvec0 = np.zeros((3, 1))
    tvec0 = np.array([[0.0], [0.0], [1500.0]])
    ok, rvec, _ = cv2.solvePnP(model, pts, k, None, rvec0, tvec0, useExtrinsicGuess=True,
                               flags=cv2.SOLVEPNP_ITERATIVE)[:3]
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    return _euler(r)
