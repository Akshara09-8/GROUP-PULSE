"""Write and read per-frame track tables. Parquet when pyarrow is installed, CSV otherwise."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrackRow:
    meeting: str
    frame: int
    t: float
    person: str
    face_found: bool
    head_yaw: float | None
    head_pitch: float | None
    gaze_dir: float | None        # top-down degrees
    gaze_target: str              # P1..P4, screen, whiteboard, table, away, unknown
    target_off: float | None
    torso_yaw: float | None       # camera-relative degrees, room view
    torso_conf: float
    low_confidence: bool
    # Direct face cues (more stable than PnP angles for pitch and large turns); None when no face.
    nose_dx: float | None = None   # nose x offset from eye midpoint / eye distance (+ = image right)
    nose_dy: float | None = None   # nose y below eye line / eye-to-mouth height (larger = head down)
    face_cx: float | None = None   # face box centre x / image width
    face_cy: float | None = None   # face box centre y / image height
    face_w: float | None = None    # face box width / image width
    eye_h: float | None = None     # eye direction from MediaPipe blendshapes, + = image right
    eye_v: float | None = None     # + = up, - = down (catches glances at the table)
    face_model: str = ""           # "yunet", "mediapipe" or "hybrid"
    torso_dir: float | None = None # top-down body direction (degrees), from torso_yaw via the close-up camera


COLUMNS = [f.name for f in fields(TrackRow)]


def write_tracks(rows: list[TrackRow], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in rows]
    if p.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(data), p, compression="zstd")
        return p
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    return p


def _parse(name: str, v: str):
    if v == "":
        return "" if name == "face_model" else None
    if name in ("frame",):
        return int(v)
    if name in ("face_found", "low_confidence"):
        return v == "True"
    if name in ("meeting", "person", "gaze_target", "face_model"):
        return v
    return float(v)


def read_tracks(path: str | Path) -> list[TrackRow]:
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [TrackRow(**r) for r in pq.read_table(p).to_pylist()]
    with p.open(newline="", encoding="utf-8") as f:
        return [TrackRow(**{k: _parse(k, v) for k, v in row.items()}) for row in csv.DictReader(f)]
