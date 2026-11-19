"""Phase 1 perception: who is where, where they look, which way they face.

Pure-geometry parts (head pose from landmarks, torso orientation, gaze targets, tracking,
metrics) need only NumPy/OpenCV. Detector adapters in `backends.py` load heavier models
(Ultralytics YOLO-pose, OpenCV YuNet) only when used.
"""

from .geometry import angle_diff, bin8, bin8_name, wrap180
from .headpose import head_pose, project_model
from .torso import torso_yaw, frontal_width
from .gaze import GazeResult, gaze_target
from .layout import SeatLayout, load_layout
from .tracker import IoUTracker, iou

__all__ = [
    "angle_diff", "bin8", "bin8_name", "wrap180",
    "head_pose", "project_model",
    "torso_yaw", "frontal_width",
    "GazeResult", "gaze_target",
    "SeatLayout", "load_layout",
    "IoUTracker", "iou",
]
