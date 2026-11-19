"""Make the hand-label set for Phase 1 tests T1.3-T1.5 (200 frames by default).

    python -m grouppulse.perception.labels --data path/to/ami-data --meetings demo --n 200

Writes data/labels/frames/*.jpg (close-up frames) and data/labels/labels.csv with empty columns
to fill in by hand: gaze_target (P1-P4, screen, whiteboard, table, away), head_dir and torso_dir
(8-way bin names from geometry.BIN8_NAMES, as seen in the image: "right" = facing image right) and
visible (yes/partly/no). Frames are spread evenly through each meeting.

Then `--sheets` makes numbered contact sheets for labelling, and `--score` runs the face and pose
models on the same frames and reports T1.1 (face found when visible), T1.3 (head direction bin)
and T1.4 (torso direction bin, target >= 80%).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..data.config import REPO_ROOT, load_config
from ..data.meeting import video_path
from .layout import layout_for_meeting

FIELDS = ["meeting", "frame", "t", "person", "image", "gaze_target", "head_dir", "torso_dir", "visible"]


def sample_frames(total_frames: int, n: int) -> list[int]:
    if n <= 0 or total_frames <= 0:
        return []
    step = total_frames / n
    return sorted({int(step * (k + 0.5)) for k in range(n)})


def contact_sheets(out: Path, per_sheet: int = 20, cols: int = 5) -> list[Path]:
    """Numbered tiles (row number in labels.csv) for labelling many frames at a glance."""
    import cv2
    import numpy as np

    with (out / "labels.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    paths = []
    for s in range(0, len(rows), per_sheet):
        tiles = []
        for k, r in enumerate(rows[s:s + per_sheet], start=s):
            img = cv2.resize(cv2.imread(str(out / r["image"])), (264, 216))
            cv2.rectangle(img, (0, 0), (52, 24), (0, 0, 0), -1)
            cv2.putText(img, str(k), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = out / f"sheet_{s // per_sheet:02d}.jpg"
        cv2.imwrite(str(p), sheet)
        paths.append(p)
    return paths


def predict(out: Path) -> list[dict]:
    """Run the hybrid face reader and YOLO-pose on every labelled frame (predictions.csv)."""
    import cv2

    from .backends import HybridFace, MeshFace, YoloPose
    from .geometry import bin8_name
    from .headpose import head_pose
    from .torso import frontal_width, signed_shoulder_width, torso_yaw

    with (out / "labels.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    face, pose = HybridFace(), YoloPose(imgsz=352)
    found = []
    for r in rows:
        img = cv2.imread(str(out / r["image"]))
        h, w = img.shape[:2]
        f = face.largest(img)
        hp = None
        if f is not None:
            hp = head_pose(f.landmarks6 if isinstance(f, MeshFace) else f.landmarks, w, h)
        found.append((r, f is not None, hp, pose.largest(img)))
    # frontal shoulder width per person from all their sampled frames
    ref = {}
    for key in {(r["meeting"], r["person"]) for r in rows}:
        ref[key] = frontal_width(signed_shoulder_width(p.keypoints) for r, _, _, p in found
                                 if p is not None and (r["meeting"], r["person"]) == key)
    preds = []
    for r, has_face, hp, p in found:
        tyaw = None
        if p is not None and ref.get((r["meeting"], r["person"])):
            tyaw, _ = torso_yaw(p.keypoints, ref[(r["meeting"], r["person"])])
        preds.append({"row": r["image"], "face_found": has_face,
                      "head_yaw": None if hp is None else round(hp[0], 1),
                      "head_dir": "" if hp is None else bin8_name(hp[0]),
                      "torso_yaw": None if tyaw is None else round(tyaw, 1),
                      "torso_dir": "" if tyaw is None else bin8_name(tyaw)})
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)
    return preds


def score(out: Path) -> dict[str, float]:
    """Compare predictions.csv with the filled-in labels.csv."""
    from .geometry import BIN8_NAMES

    with (out / "labels.csv").open(encoding="utf-8") as fh:
        labels = {r["image"]: r for r in csv.DictReader(fh)}
    with (out / "predictions.csv").open(encoding="utf-8") as fh:
        preds = {r["row"]: r for r in csv.DictReader(fh)}

    def near(a: str, b: str) -> bool:     # same bin or a neighbour (within ~45 degrees)
        i, j = BIN8_NAMES.index(a), BIN8_NAMES.index(b)
        return min((i - j) % 8, (j - i) % 8) <= 1

    res: dict[str, float] = {}
    vis = [k for k, l in labels.items() if l["visible"] in ("yes", "partly")]
    res["T1.1 face found when visible"] = sum(preds[k]["face_found"] == "True" for k in vis) / len(vis)
    absent = [k for k, l in labels.items() if l["visible"] == "no"]
    if absent:
        res["  face 'found' when nobody there"] = sum(preds[k]["face_found"] == "True" for k in absent) / len(absent)
    for key, name in (("head_dir", "T1.3 head"), ("torso_dir", "T1.4 torso")):
        pairs = [(labels[k][key], preds[k][key]) for k in labels if labels[k][key] and preds[k][key]]
        missing = sum(1 for k in labels if labels[k][key] and not preds[k][key])
        res[f"{name} bin exact"] = sum(a == b for a, b in pairs) / len(pairs)
        res[f"{name} bin within one"] = sum(near(a, b) for a, b in pairs) / len(pairs)
        res[f"  {name} frames scored / no prediction"] = float(len(pairs)), float(missing)
    return res


def main(argv: list[str] | None = None) -> int:
    import cv2

    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meetings", nargs="+", default=["demo"])
    ap.add_argument("--n", type=int, default=200, help="total frames across meetings and people")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "labels")
    ap.add_argument("--sheets", action="store_true", help="make numbered contact sheets for labelling")
    ap.add_argument("--predict", action="store_true", help="run the models on the sampled frames")
    ap.add_argument("--score", action="store_true", help="score predictions against filled-in labels")
    args = ap.parse_args(argv)

    if args.sheets:
        print("\n".join(str(p) for p in contact_sheets(args.out)))
        return 0
    if args.predict:
        print(f"{len(predict(args.out))} predictions -> {args.out / 'predictions.csv'}")
        return 0
    if args.score:
        for k, v in score(args.out).items():
            print(f"{k:<40} {v if isinstance(v, tuple) else f'{v:.1%}'}")
        return 0

    meetings = cfg.meetings(args.meetings)
    jobs = [(m, p, v) for m in meetings for p, v in layout_for_meeting(m).camera_of.items()]
    per_job = max(1, args.n // max(1, len(jobs)))
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for m, person, view in jobs:
        path = video_path(args.data, m, view)
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in sample_frames(total, per_job):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            name = f"{m}_{person}_{f:06d}.jpg"
            cv2.imwrite(str(args.out / "frames" / name), img)
            rows.append({"meeting": m, "frame": f, "t": round(f / cfg.fps, 2), "person": person,
                         "image": f"frames/{name}", "gaze_target": "", "head_dir": "", "torso_dir": "",
                         "visible": ""})
        cap.release()
    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} frames to label -> {args.out / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
