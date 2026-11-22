"""Single-camera perception for uploads: one fixed camera filming the whole group (EXPERIMENTAL).

The AMI pipeline (run.py) has one close-up per person and a calibrated gaze model per camera.
An upload has one wide view and no calibration, so this path uses a plain geometric rule in the
image plane, and its gaze targets are not validated against labels:

  1. Faces: YuNet finds every face; an IoU tracker links them over time; the `people` most
     persistent tracks are kept and named P1..PN from left to right.
  2. Head pose: MediaPipe mesh on a crop around each face (6-point PnP), else YuNet's 5 points.
  3. Gaze target: pitch <= -20 deg -> "table"; else the other person whose head lies within a
     35 deg cone of the head's turn direction in the image; else "away". Known limitation: the
     image has no depth, so the head closest to the gaze *line* wins, and someone further along
     that line can be chosen over a nearer person.
  4. Torso: YOLO-pose at 1 fps; each face takes the body whose box contains it.

Output: TrackRows for grouppulse.social.analyze, plus an overlay (per-frame face boxes and gaze
targets, coordinates as fractions of the frame) that the web app draws over the video.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .geometry import angle_diff
from .headpose import head_pose
from .io import TrackRow
from .torso import frontal_width, signed_shoulder_width, torso_yaw
from .tracker import IoUTracker

TABLE_PITCH = -20.0
CONE_DEG = 35.0
MIN_TURN = 0.15          # |sin(yaw)| below this (~9 deg) = facing the camera: no horizontal direction
DETECT_WIDTH = 960       # frames are scaled to this width for detection (keeps 1080p uploads fast)


@dataclass
class FaceObs:
    frame: int
    t: float
    track: int
    box: tuple[float, float, float, float]      # pixels in the scaled frame
    yaw: float | None
    pitch: float | None


def image_gaze_target(viewer: str, heads: dict[str, tuple[float, float]], yaw: float | None,
                      pitch: float | None) -> str:
    """Target from head pose and head positions in one image (x right, y down).

    The head's turn gives a horizontal direction (+yaw = towards image right) and its pitch a small
    vertical component; the nearest other head inside the cone wins."""
    if yaw is None or pitch is None:
        return "unknown"
    if pitch <= TABLE_PITCH:
        return "table"
    sx = math.sin(math.radians(yaw))
    if abs(sx) < MIN_TURN:
        return "away"
    gx, gy = sx, -0.5 * math.sin(math.radians(pitch))
    g = math.degrees(math.atan2(gy, gx))
    vx, vy = heads[viewer]
    best, best_off = "away", None
    for other, (x, y) in heads.items():
        if other == viewer:
            continue
        off = angle_diff(g, math.degrees(math.atan2(y - vy, x - vx)))
        if off <= CONE_DEG and (best_off is None or off < best_off):
            best, best_off = other, off
    return best


def name_tracks(obs: list[FaceObs], people: int) -> dict[int, str]:
    """Keep the `people` tracks seen most often; name them P1..PN by mean x, left to right."""
    counts = Counter(o.track for o in obs)
    keep = [tid for tid, _ in counts.most_common(people)]
    mean_x = {tid: sum((o.box[0] + o.box[2]) / 2 for o in obs if o.track == tid) / counts[tid] for tid in keep}
    return {tid: f"P{i + 1}" for i, tid in enumerate(sorted(keep, key=mean_x.get))}


def assign_people(obs: list[FaceObs], people: int) -> dict[int, dict[str, FaceObs]]:
    """frame -> {person: face}. Seats are fixed in a meeting, so tracks only locate the seats:
    the `people` most persistent tracks give seat centres, then every detection (including the
    short fragments a tracker leaves when someone turns away) goes to the nearest seat, if it is
    closer than half the gap between neighbouring seats. One face per seat per frame."""
    names = name_tracks(obs, people)
    centre = {}
    for tid, p in names.items():
        mine = [o for o in obs if o.track == tid]
        centre[p] = (sum((o.box[0] + o.box[2]) / 2 for o in mine) / len(mine),
                     sum((o.box[1] + o.box[3]) / 2 for o in mine) / len(mine))
    pts = list(centre.values())
    gaps = [math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]]
    reach = 0.5 * min(gaps) if gaps else float("inf")
    out: dict[int, dict[str, FaceObs]] = defaultdict(dict)
    best_d: dict[tuple[int, str], float] = {}
    for o in obs:
        c = ((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2)
        p, d = min(((p, math.dist(c, xy)) for p, xy in centre.items()), key=lambda x: x[1], default=(None, 0))
        if p is None or d > reach:
            continue
        if d < best_d.get((o.frame, p), float("inf")):
            best_d[(o.frame, p)] = d
            out[o.frame][p] = o
    return out


def assemble(obs: list[FaceObs], people: int, size: tuple[int, int], fps: float, every: int,
             torso: dict[tuple[int, str], tuple[float | None, float]] | None = None,
             session: str = "upload") -> tuple[list[TrackRow], dict]:
    """Face observations -> TrackRows (every person at every sampled frame) and the overlay."""
    w, h = size
    by_frame = assign_people(obs, people)
    labels = sorted(name_tracks(obs, people).values(), key=lambda p: int(p[1:]))
    frames = sorted({o.frame for o in obs} | set(by_frame))
    last_pos: dict[str, tuple[float, float]] = {}
    rows, overlay_frames = [], []
    for f in frames:
        seen = by_frame.get(f, {})
        for p, o in seen.items():
            last_pos[p] = ((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2)
        entry = {"t": round(f / fps, 2), "people": {}}
        for p in labels:
            o = seen.get(p)
            target = image_gaze_target(p, last_pos, o.yaw, o.pitch) if o is not None and p in last_pos else "unknown"
            ty, tc = (torso or {}).get((f, p), (None, 0.0))
            rows.append(TrackRow(session, f, round(f / fps, 3), p, o is not None,
                                 None if o is None or o.yaw is None else round(o.yaw, 1),
                                 None if o is None or o.pitch is None else round(o.pitch, 1),
                                 None, target, None, ty, tc, o is None, face_model="single-camera"))
            if o is not None:
                entry["people"][p] = {"box": [round(o.box[0] / w, 4), round(o.box[1] / h, 4),
                                              round(o.box[2] / w, 4), round(o.box[3] / h, 4)], "target": target}
        overlay_frames.append(entry)
    return rows, {"fps": fps / every, "people": labels, "frames": overlay_frames}


def _frames(path: Path, every: int, limit: int | None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    try:
        while True:
            if not cap.grab() or (limit is not None and i >= limit):
                break
            if i % every == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i, fps, frame
            i += 1
    finally:
        cap.release()


def process_video(path: str | Path, people: int, sample_fps: float = 2.5, max_seconds: float | None = 600,
                  with_torso: bool = True, session: str = "upload") -> tuple[list[TrackRow], dict]:
    import cv2

    from .backends import MediaPipeFace, MeshFace, YuNetFace

    finder = YuNetFace(score_threshold=0.5)
    try:
        reader = MediaPipeFace(max_faces=1)
    except Exception:            # mediapipe missing: fall back to YuNet's 5 points
        reader = None
    pose = None
    if with_torso:
        try:
            from .backends import YoloPose
            pose = YoloPose(imgsz=416)   # 640 took 1.6 s a frame on CPU
            import numpy as np
            for _ in range(3):           # the first calls are several times slower (warm-up)
                pose.detect(np.zeros((288, 352, 3), dtype=np.uint8))
        except Exception:
            pose = None

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    every = max(1, round(fps / sample_fps))
    limit = int(max_seconds * fps) if max_seconds else None
    tracker = IoUTracker(min_iou=0.2, max_misses=25)
    obs: list[FaceObs] = []
    bodies: dict[int, list] = {}
    size = (0, 0)
    for i, _, frame in _frames(Path(path), every, limit):
        scale = min(1.0, DETECT_WIDTH / frame.shape[1])
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        size = (w, h)
        faces = finder.detect(frame)
        ids = tracker.update([f.box for f in faces])
        for face, tid in zip(faces, ids):
            mesh = reader.on_crop(frame, face.box) if reader is not None else None
            hp = head_pose(mesh.landmarks6 if isinstance(mesh, MeshFace) else face.landmarks, w, h)
            obs.append(FaceObs(i, i / fps, tid, face.box, hp[0] if hp else None, hp[1] if hp else None))
        if pose is not None and i % max(every, round(fps)) == 0:
            bodies[i] = pose.detect(frame)

    torso = _torso_by_person(obs, bodies, people, every, fps) if bodies else None
    return assemble(obs, people, size, fps, every, torso, session)


def _torso_by_person(obs, bodies, people, every, fps):
    """Match each named face to the body box containing its centre; hold readings for one second."""
    face_at = assign_people(obs, people)
    widths, matched = defaultdict(list), {}
    for f, people_bodies in bodies.items():
        for p, o in face_at.get(f, {}).items():
            cx, cy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
            body = next((b for b in people_bodies if b.box[0] <= cx <= b.box[2] and b.box[1] <= cy <= b.box[3]), None)
            if body is not None:
                matched[(f, p)] = body
                widths[p].append(signed_shoulder_width(body.keypoints))
    ref = {p: frontal_width(ws) for p, ws in widths.items()}
    out = {}
    hold = max(every, round(fps))
    for (f, p), body in matched.items():
        if ref.get(p):
            reading = torso_yaw(body.keypoints, ref[p])
            for g in range(f, f + hold, every):
                out[(g, p)] = (None if reading[0] is None else round(reading[0], 1), round(reading[1], 2))
    return out
