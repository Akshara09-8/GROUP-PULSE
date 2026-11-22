"""Model adapters. Each loads its model only when constructed.

- YuNetFace: OpenCV's YuNet face detector (ONNX, ~0.2 MB). Runs on CPU, ships with opencv-python.
  Gives a box and 5 landmarks per face, which headpose.head_pose turns into yaw/pitch.
- YoloPose: Ultralytics YOLO-pose for body keypoints. Used on each close-up (shoulders are in
  frame and there is one person per camera) and optionally on the Idiap room camera with ByteTrack.
  Optional: `pip install -e .[pose]` (AGPL-3.0; fine for a free demo, check before commercial use).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    landmarks: np.ndarray                     # (5, 2): eye img-left, eye img-right, nose, mouth img-left, mouth img-right
    score: float


@dataclass(frozen=True)
class PersonPose:
    track_id: int | None
    box: tuple[float, float, float, float]
    keypoints: np.ndarray                     # (17, 3) COCO x, y, confidence
    score: float


def fetch_yunet(dest: Path | None = None) -> Path:
    dest = dest or MODELS_DIR / Path(YUNET_URL).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, dest)
    return dest


def face_cues(face: Face, width: int, height: int) -> dict[str, float]:
    """Scale-free cues from YuNet's 5 points and box (see io.TrackRow for meanings)."""
    lm = face.landmarks
    eye_mid, mouth_mid = (lm[0] + lm[1]) / 2, (lm[3] + lm[4]) / 2
    eye_dist = float(np.linalg.norm(lm[1] - lm[0])) or 1.0
    face_h = float(mouth_mid[1] - eye_mid[1]) or 1.0
    x1, y1, x2, y2 = face.box
    return {
        "nose_dx": round(float(lm[2, 0] - eye_mid[0]) / eye_dist, 4),
        "nose_dy": round(float(lm[2, 1] - eye_mid[1]) / face_h, 4),
        "face_cx": round((x1 + x2) / 2 / width, 4),
        "face_cy": round((y1 + y2) / 2 / height, 4),
        "face_w": round((x2 - x1) / width, 4),
    }


class YuNetFace:
    def __init__(self, model_path: str | Path | None = None, score_threshold: float = 0.6):
        import cv2

        path = Path(model_path) if model_path else fetch_yunet()
        self._cv2 = cv2
        self._det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, 0.3, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(frame)
        out: list[Face] = []
        for f in faces if faces is not None else []:
            x, y, bw, bh = (float(v) for v in f[:4])
            out.append(Face((x, y, x + bw, y + bh), np.asarray(f[4:14], dtype=float).reshape(5, 2), float(f[14])))
        return out

    def largest(self, frame: np.ndarray) -> Face | None:
        faces = self.detect(frame)
        return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), default=None)


MP_FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
               "face_landmarker/float16/1/face_landmarker.task")
# Face-mesh indices for the 6-point head-pose model, in headpose.MODEL_6 order (by image side):
# outer eye corner image-left (person's right, 33), image-right (263), nose tip (1),
# mouth corner image-left (61), image-right (291), chin (152).
MESH_6 = (33, 263, 1, 61, 291, 152)


@dataclass(frozen=True)
class MeshFace:
    box: tuple[float, float, float, float]
    landmarks6: np.ndarray            # (6, 2) pixels, MODEL_6 order
    eye_h: float                      # eye direction, + = toward image right (blendshape units, -1..1)
    eye_v: float                      # + = up, - = down
    score: float = 1.0


def eye_direction(bs: dict[str, float]) -> tuple[float, float]:
    """Eye gaze from MediaPipe blendshapes, independent of head pose.

    'Left'/'Right' in blendshape names are the person's. Looking toward image right means the
    person's left eye looks out and their right eye looks in."""
    h = ((bs.get("eyeLookOutLeft", 0) - bs.get("eyeLookInLeft", 0))
         + (bs.get("eyeLookInRight", 0) - bs.get("eyeLookOutRight", 0))) / 2
    v = ((bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0))
         - (bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0))) / 2
    return round(float(h), 4), round(float(v), 4)


class MediaPipeFace:
    """MediaPipe Face Landmarker: 478-point mesh + 52 blendshapes per face, CPU.

    Gives a 6-point head pose (chin and eye corners, which YuNet lacks) and eye-direction cues."""

    def __init__(self, model_path: str | Path | None = None, max_faces: int = 2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        path = Path(model_path) if model_path else self._fetch()
        self._mp = mp
        self._lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.IMAGE, num_faces=max_faces,
            output_face_blendshapes=True, min_face_detection_confidence=0.4,
        ))

    @staticmethod
    def _fetch() -> Path:
        dest = MODELS_DIR / "face_landmarker.task"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MP_FACE_URL, dest)
        return dest

    def detect(self, frame_bgr: np.ndarray) -> list[MeshFace]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out = []
        for i, pts in enumerate(res.face_landmarks or []):
            xy = np.array([[p.x * w, p.y * h] for p in pts])
            bs = {c.category_name: c.score for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            eh, ev = eye_direction(bs)
            x1, y1 = xy.min(axis=0)
            x2, y2 = xy.max(axis=0)
            out.append(MeshFace((float(x1), float(y1), float(x2), float(y2)), xy[list(MESH_6)], eh, ev))
        return out

    def largest(self, frame: np.ndarray) -> MeshFace | None:
        faces = self.detect(frame)
        return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), default=None)

    def on_crop(self, frame: np.ndarray, box: tuple[float, float, float, float],
                margin: float = 0.6, size: int = 320) -> MeshFace | None:
        """Run the mesh on an enlarged crop around a face box from another detector.

        MediaPipe's own short-range detector misses turned or small faces in AMI close-ups;
        cropping around YuNet's box fixes most of those. Landmarks are mapped back to frame pixels."""
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if c - a < 8 or d - b < 8:
            return None
        crop = frame[b:d, a:c]
        scale = size / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        face = self.largest(crop)
        if face is None:
            return None
        lm = face.landmarks6 / scale + np.array([a, b])
        fx1, fy1, fx2, fy2 = face.box
        return MeshFace((fx1 / scale + a, fy1 / scale + b, fx2 / scale + a, fy2 / scale + b),
                        lm, face.eye_h, face.eye_v)

    def close(self) -> None:
        self._lm.close()


class HybridFace:
    """YuNet finds the face (robust to turns and size); MediaPipe reads it (dense mesh + eyes).

    Returns a MeshFace when the mesh succeeds, else the YuNet Face so no frame is lost."""

    def __init__(self):
        self.finder, self.reader = YuNetFace(), MediaPipeFace(max_faces=1)

    def largest(self, frame: np.ndarray):
        found = self.finder.largest(frame)
        if found is None:
            return None
        return self.reader.on_crop(frame, found.box) or found


YOLO_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"


def fetch_yolo_pose(dest: Path | None = None) -> Path:
    """Download YOLO-pose weights with urllib (uses the Windows certificate store).

    Ultralytics' own downloader uses `requests` + certifi, which fails certificate checks on
    machines whose antivirus or proxy re-signs HTTPS; fetching here avoids that without turning
    verification off."""
    dest = dest or MODELS_DIR / Path(YOLO_POSE_URL).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YOLO_POSE_URL, dest)
    return dest


class YoloPose:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.3, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Body orientation needs Ultralytics: pip install -e .[pose]") from e
        self._model = YOLO(str(weights or fetch_yolo_pose()))
        self._conf, self._imgsz = conf, imgsz

    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        """Every person in the frame, no tracking (avoids ByteTrack's extra runtime dependency)."""
        r = self._model(frame, imgsz=self._imgsz, conf=self._conf, verbose=False)[0]
        if r.boxes is None or not len(r.boxes) or r.keypoints is None:
            return []
        kps = r.keypoints.data.cpu().numpy()
        return [PersonPose(None, tuple(float(v) for v in b), k, float(s))
                for b, k, s in zip(r.boxes.xyxy.cpu().numpy(), kps, r.boxes.conf.cpu().numpy())]

    def largest(self, frame: np.ndarray) -> PersonPose | None:
        """The biggest person in the frame (the close-up's subject, not someone walking behind)."""
        r = self._model(frame, imgsz=self._imgsz, conf=self._conf, verbose=False)[0]
        if r.boxes is None or not len(r.boxes) or r.keypoints is None:
            return None
        boxes = r.boxes.xyxy.cpu().numpy()
        i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        return PersonPose(None, tuple(float(v) for v in boxes[i]), r.keypoints.data.cpu().numpy()[i],
                          float(r.boxes.conf[i]))

    def track(self, frame: np.ndarray) -> list[PersonPose]:
        r = self._model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self._conf,
                              imgsz=self._imgsz, verbose=False)[0]
        if r.boxes is None or r.keypoints is None:
            return []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        kps = r.keypoints.data.cpu().numpy()
        return [
            PersonPose(tid, tuple(float(v) for v in box), kp, float(s))
            for tid, box, kp, s in zip(ids, r.boxes.xyxy.cpu().numpy(), kps, r.boxes.conf.cpu().numpy())
        ]
