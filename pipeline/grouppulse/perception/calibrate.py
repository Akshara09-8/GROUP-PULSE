"""Calibrate gaze-target mapping on meetings with AMI focus-of-attention labels.

    # 1. head pose for a labelled meeting (close-ups only)
    python -m grouppulse.perception.run --data D --meeting ES2008a --no-torso --gaze-model geometry
    # 2. fit + validate, write configs/vfoa_<site>.toml
    python -m grouppulse.perception.calibrate --data D --meetings ES2008a --site edinburgh

Validation: with one meeting, fit on one half and test on the other, both ways round (by time);
with several, leave one meeting out. Reports, on the held-out part:
  majority-class baseline, geometric layout as configured, geometric with refitted camera
  directions, learned VFOA (raw and 1 s smoothed).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..data.annotations import find_ann_root, load_focus, load_speakers, seat_of_speaker
from ..data.config import REPO_ROOT
from .evaluate import truth_at
from .gaze import gaze_target
from .io import TrackRow, read_tracks
from .layout import SeatLayout, layout_for_meeting
from .metrics import target_accuracy
from .vfoa import FEATURE_SETS, Sample, viterbi, default_path, fit, fit_camera_dirs, smooth


def labelled_rows(data: Path, meeting: str, tracks_dir: Path) -> list[tuple[TrackRow, str]]:
    """Track rows paired with their true target (in seat names); rows without a label dropped."""
    ann = find_ann_root(data / "annotations")
    seat = seat_of_speaker(load_speakers(ann, meeting))
    spans: dict[str, list] = {}
    for f in load_focus(ann, meeting):
        if f.speaker in seat:
            spans.setdefault(seat[f.speaker], []).append(f)
    out = []
    for r in read_tracks(tracks_dir / f"{meeting}.csv"):
        s = truth_at(spans.get(r.person, []), r.t)
        if s is not None:
            out.append((r, seat.get(s.target, s.target)))
    return out


def to_samples(pairs, layout: SeatLayout, features: tuple[str, ...] = FEATURE_SETS["angles"]) -> list[Sample]:
    """Samples for the VFOA fit. Rows missing an extra feature keep yaw/pitch only (used by the fallback)."""
    out = []
    for r, t in pairs:
        if r.head_yaw is None or r.person not in layout.camera_of:
            continue
        extra = tuple(getattr(r, f, None) for f in features[2:])
        if any(v is None for v in extra):
            extra = ()
        out.append(Sample(layout.camera_of[r.person], r.head_yaw, r.head_pitch, t, extra))
    return out


def _predict(model, layout, test, features):
    out = []
    for r, _ in test:
        view = layout.camera_of[r.person]
        extra = tuple(getattr(r, f, None) for f in features[2:])
        out.append(model.predict(view, r.head_yaw, r.head_pitch, () if any(v is None for v in extra) else extra))
    return out


def _smooth_by_person(test, labels):
    out = list(labels)
    for person in {r.person for r, _ in test}:
        idx = [i for i, (r, _) in enumerate(test) if r.person == person]
        for i, lab in zip(idx, smooth([labels[i] for i in idx], 5)):
            out[i] = lab
    return out


def _hmm_by_person(model, layout, test, features, stay: float = 0.9):
    """Sticky-HMM decoding per person, in time order (attention tends to stay put)."""
    out = ["unknown"] * len(test)
    for person in {r.person for r, _ in test}:
        idx = sorted((i for i, (r, _) in enumerate(test) if r.person == person), key=lambda i: test[i][0].t)
        frames = []
        for i in idx:
            r = test[i][0]
            extra = tuple(getattr(r, f, None) for f in features[2:])
            frames.append(model.scores(layout.camera_of[r.person], r.head_yaw, r.head_pitch,
                                       () if any(v is None for v in extra) else extra))
        for i, lab in zip(idx, viterbi(frames, stay)):
            out[i] = lab
    return out


def score(train, test, layout: SeatLayout, site: str) -> tuple[dict[str, float], dict[str, float]]:
    truth = [t for _, t in test]
    majority = max(set(t for _, t in train), key=[t for _, t in train].count)
    dirs = fit_camera_dirs(to_samples(train, layout), layout)
    refit = replace(layout, camera_dir={**layout.camera_dir, **dirs})
    geo_refit = []
    for r, _ in test:
        view = layout.camera_of[r.person]
        gdir = refit.world_gaze(view, r.head_yaw) if r.head_yaw is not None else None
        geo_refit.append(gaze_target(refit, r.person, gdir, r.head_pitch).target)
    res = {
        "majority": target_accuracy(truth, [majority] * len(truth)),
        # whatever produced the track table's gaze_target column (geometry, or a saved model;
        # a saved model fitted on this same meeting makes this an in-sample number)
        "targets stored in tracks": target_accuracy(truth, [r.gaze_target for r, _ in test]),
        "geometry (refit cameras)": target_accuracy(truth, geo_refit),
    }
    have = lambda f: any(getattr(r, f, None) is not None for r, _ in train)  # noqa: E731
    for name, feats in FEATURE_SETS.items():
        if not all(have(f) for f in feats):
            continue
        model = fit(to_samples(train, layout, feats), site, features=feats)
        pred = _predict(model, layout, test, feats)
        res[f"vfoa {name}"] = target_accuracy(truth, pred)
        res[f"vfoa {name} + 1 s smoothing"] = target_accuracy(truth, _smooth_by_person(test, pred))
        res[f"vfoa {name} + HMM"] = target_accuracy(truth, _hmm_by_person(model, layout, test, feats))
    return res, dirs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--site", choices=["edinburgh", "idiap"], default=None)
    ap.add_argument("--tracks", type=Path, default=REPO_ROOT / "data" / "tracks")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="angles",
                    help="feature set for the saved model (eyes needs --face hybrid/mediapipe tracks)")
    args = ap.parse_args(argv)

    layout = layout_for_meeting(args.meetings[0])
    site = args.site or layout.name
    per_meeting = {m: labelled_rows(args.data, m, args.tracks) for m in args.meetings}
    folds = []
    if len(per_meeting) == 1:
        rows = sorted(next(iter(per_meeting.values())), key=lambda p: p[0].t)
        cut = rows[len(rows) // 2][0].t
        first, second = [p for p in rows if p[0].t < cut], [p for p in rows if p[0].t >= cut]
        # both directions: one split alone can be lucky (ES2008a: 68% one way, 53% the other)
        folds.append(("first half -> second half", first, second))
        folds.append(("second half -> first half", second, first))
    else:
        for held in per_meeting:
            train = [p for m, rs in per_meeting.items() if m != held for p in rs]
            folds.append((f"held out {held}", train, per_meeting[held]))

    totals: dict[str, list[float]] = {}
    for name, train, test in folds:
        res, dirs = score(train, test, layout, site)
        print(f"{name}: {len(train)} train / {len(test)} test frames; refit camera_dir {dirs}")
        for k, v in res.items():
            print(f"  {k:<26} {v:6.1%}")
            totals.setdefault(k, []).append(v)
    if len(folds) > 1:
        print("mean over folds:")
        for k, v in totals.items():
            print(f"  {k:<26} {sum(v) / len(v):6.1%}")

    if not args.no_save:
        all_rows = [p for rs in per_meeting.values() for p in rs]
        feats = FEATURE_SETS[args.features]
        path = fit(to_samples(all_rows, layout, feats), site, features=feats).save(default_path(site))
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
