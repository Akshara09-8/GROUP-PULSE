"""Run Phase 1 perception on one AMI meeting and write a per-frame table.

    python -m grouppulse.perception.run --data path/to/ami-data --meeting ES2002a --fps 5 --max-seconds 120

For each person (P1-P4) and each sampled frame:
  close-up view -> face (YuNet) -> head yaw/pitch -> top-down gaze direction -> gaze target
  room view     -> people + keypoints (YOLO-pose, optional) -> torso yaw per tracked person
Room-view tracks are matched to seats left-to-right; this is a first pass to calibrate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..data.config import REPO_ROOT, load_config
from ..data.meeting import video_path
from .gaze import UNKNOWN, gaze_target
from .headpose import head_pose
from .io import TrackRow, write_tracks
from .layout import layout_for_meeting
from .torso import frontal_width, signed_shoulder_width, torso_yaw


def _frames(path: Path, every: int, limit: int | None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok or (limit is not None and i >= limit):
                break
            if i % every == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i, frame
            i += 1
    finally:
        cap.release()


def pick_face_model(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import mediapipe  # noqa: F401
        return "hybrid"
    except ImportError:
        return "yunet"


def _face_reading(det, kind: str, frame, w: int, h: int):
    """(found, yaw, pitch, low_conf, extra cue columns) for one close-up frame."""
    from .backends import face_cues

    from .backends import MeshFace

    face = det.largest(frame)
    if face is None:
        return False, None, None, True, {}
    if isinstance(face, MeshFace):
        pose = head_pose(face.landmarks6, w, h)
        x1, y1, x2, y2 = face.box
        cues = {"eye_h": face.eye_h, "eye_v": face.eye_v,
                "face_cx": round((x1 + x2) / 2 / w, 4), "face_cy": round((y1 + y2) / 2 / h, 4),
                "face_w": round((x2 - x1) / w, 4)}
        low = False
    else:
        pose = head_pose(face.landmarks, w, h)
        cues = face_cues(face, w, h)
        low = face.score < 0.8
    yaw, pitch = pose if pose else (None, None)
    return True, yaw, pitch, low, cues


def gaze_rows(data: Path, meeting: str, every: int, limit: int | None, fps: float,
              gaze_model: str = "auto", face_model: str = "auto") -> dict[tuple[int, str], TrackRow]:
    from .backends import HybridFace, MediaPipeFace, YuNetFace
    from .vfoa import VfoaModel, default_path

    layout = layout_for_meeting(meeting)
    vfoa = None
    if gaze_model in ("auto", "vfoa") and default_path(layout.name).exists():
        vfoa = VfoaModel.load(default_path(layout.name))
        print(f"  gaze targets: learned model {default_path(layout.name).name}")
    elif gaze_model == "vfoa":
        raise FileNotFoundError(f"No calibrated model at {default_path(layout.name)}; run calibrate first")
    else:
        print("  gaze targets: geometric layout (uncalibrated)")
    kind = pick_face_model(face_model)
    print(f"  face model: {kind}")
    rows: dict[tuple[int, str], TrackRow] = {}
    for person, view in layout.camera_of.items():
        path = video_path(data, meeting, view)
        if not path.exists():
            print(f"  skip {view}: not downloaded")
            continue
        det = {"mediapipe": MediaPipeFace, "hybrid": HybridFace}.get(kind, YuNetFace)()  # fresh per camera
        for i, frame in _frames(path, every, limit):
            h, w = frame.shape[:2]
            found, yaw, pitch, low, cues = _face_reading(det, kind, frame, w, h)
            gdir = layout.world_gaze(view, yaw) if yaw is not None else None
            g = gaze_target(layout, person, gdir, pitch)
            row = TrackRow(
                meeting, i, round(i / fps, 3), person, found,
                None if yaw is None else round(yaw, 1), None if pitch is None else round(pitch, 1),
                None if gdir is None else round(gdir, 1), g.target,
                None if g.off_angle is None else round(g.off_angle, 1), None, 0.0,
                low, **cues, face_model=kind,
            )
            if vfoa is not None:
                row.gaze_target, row.target_off = vfoa.predict_row(view, row), None
            rows[(i, person)] = row
    return rows


def _set_torso(row: TrackRow, yaw: float | None, conf: float, view: str | None, layout) -> None:
    row.torso_yaw = None if yaw is None else round(yaw, 1)
    row.torso_conf = round(conf, 2)
    if yaw is not None and view is not None:
        row.torso_dir = round(layout.world_gaze(view, yaw), 1)   # same camera geometry as head pose
    row.low_confidence = row.low_confidence or conf < 0.5


def add_torso_closeups(rows: dict[tuple[int, str], TrackRow], data: Path, meeting: str,
                       every: int, limit: int | None, pose, torso_every: int) -> None:
    """Torso direction from each person's own close-up (default).

    Close-ups frame head and shoulders with one person per camera, so no seat matching is needed.
    The frontal shoulder width is learned per person over the whole run (torso.frontal_width).
    Bodies turn slowly, so pose runs every `torso_every` frames and each reading is held until
    the next one (keeps the pipeline inside T1.9's 2x real-time budget on CPU)."""
    layout = layout_for_meeting(meeting)
    for person, view in layout.camera_of.items():
        path = video_path(data, meeting, view)
        if not path.exists():
            continue
        seen = [(i, pose.largest(frame)) for i, frame in _frames(path, torso_every, limit)]
        ref = frontal_width(signed_shoulder_width(p.keypoints) for _, p in seen if p is not None)
        for i, p in seen:
            reading = (None, 0.0) if p is None or not ref else torso_yaw(p.keypoints, ref)
            for j in range(i, i + torso_every, every):
                if (j, person) in rows:
                    _set_torso(rows[(j, person)], reading[0], reading[1],
                               view if reading[0] is not None else None, layout)


def add_torso(rows: dict[tuple[int, str], TrackRow], data: Path, meeting: str, room_view: str,
              every: int, limit: int | None, source: str = "closeup", torso_every: int | None = None) -> None:
    try:
        from .backends import YoloPose

        pose = YoloPose(imgsz=352)   # AMI video is 352x288; upscaling to 640 only costs time
    except ImportError as e:
        print(f"  skip torso: {e}")
        return
    torso_every = max(every, (torso_every or every) // every * every)
    print(f"  torso source: {source}, every {torso_every} frames")
    if source == "closeup":
        add_torso_closeups(rows, data, meeting, every, limit, pose, torso_every)
        return
    # Room camera + tracking. Works on the Idiap 'C' view; the Edinburgh fisheye 'Corner' view
    # shows people too small to find reliably (0-2 of 4 in tests), so it is not the default.
    path = video_path(data, meeting, room_view)
    if not path.exists():
        print(f"  skip torso: {room_view} not downloaded")
        return
    seen: dict[int, list] = {}
    frames: list[tuple[int, list]] = []
    for i, frame in _frames(path, every, limit):
        people = pose.track(frame)
        frames.append((i, people))
        for p in people:
            if p.track_id is not None:
                seen.setdefault(p.track_id, []).append(signed_shoulder_width(p.keypoints))
    ref = {tid: frontal_width(ws) for tid, ws in seen.items()}
    labels = sorted(layout_for_meeting(meeting).seats)
    for i, people in frames:
        ordered = sorted((p for p in people if p.track_id is not None), key=lambda p: p.box[0])
        for label, p in zip(labels, ordered):
            if (i, label) in rows and ref.get(p.track_id):
                yaw, conf = torso_yaw(p.keypoints, ref[p.track_id])
                _set_torso(rows[(i, label)], yaw, conf, None, layout_for_meeting(meeting))


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--fps", type=float, default=5.0, help="analysis frames per second (video is 25)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-torso", action="store_true", help="same as --torso none")
    ap.add_argument("--torso", choices=["closeup", "room", "none"], default="closeup",
                    help="closeup = shoulders in each person's close-up (default); room = room camera (Idiap only)")
    ap.add_argument("--torso-fps", type=float, default=1.0,
                    help="torso samples per second, held in between (default 1; bodies turn slowly)")
    ap.add_argument("--face", choices=["auto", "hybrid", "mediapipe", "yunet"], default="auto",
                    help="hybrid = YuNet finds the face, MediaPipe reads it (default when mediapipe is installed)")
    ap.add_argument("--gaze-model", choices=["auto", "vfoa", "geometry"], default="auto",
                    help="auto = learned VFOA model if configs/vfoa_<site>.toml exists, else geometry")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    every = max(1, round(cfg.fps / args.fps))
    limit = int(args.max_seconds * cfg.fps) if args.max_seconds else None
    t0 = time.perf_counter()
    rows = gaze_rows(args.data, args.meeting, every, limit, cfg.fps, args.gaze_model, args.face)
    if not args.no_torso and args.torso != "none":
        room = cfg.views_for(args.meeting, ["room"])[0]
        add_torso(rows, args.data, args.meeting, room, every, limit, args.torso,
                  max(1, round(cfg.fps / args.torso_fps)))
    out = args.out or REPO_ROOT / "data" / "tracks" / f"{args.meeting}.csv"
    ordered = [rows[k] for k in sorted(rows)]
    write_tracks(ordered, out)
    secs = time.perf_counter() - t0
    video_secs = (limit or max((r.frame for r in ordered), default=0)) / cfg.fps
    faces = sum(r.face_found for r in ordered)
    unknown = sum(r.gaze_target == UNKNOWN for r in ordered)
    print(f"{len(ordered)} rows -> {out}")
    print(f"faces found in {faces}/{len(ordered)} rows; gaze unknown in {unknown}")
    torso = sum(r.torso_yaw is not None for r in ordered)
    if not args.no_torso and args.torso != "none":
        print(f"torso direction in {torso}/{len(ordered)} rows")
    if video_secs:
        print(f"speed: {secs:.1f} s for {video_secs:.0f} s of video = {secs / video_secs:.2f}x real time (T1.9 limit 2x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
