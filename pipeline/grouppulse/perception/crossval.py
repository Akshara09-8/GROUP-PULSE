"""Leave-one-series-out gaze targets, so later scoring never sees a model fitted on its own people.

    python -m grouppulse.perception.crossval --data D --meetings focus --tracks D/tracks_idiap \
        --out D/tracks_idiap_loo [--extra IS1009a IS1009b]

For each labelled meeting M: fit the VFOA model on every labelled meeting from a *different*
series (different people; see data/splits.py), rewrite M's
gaze_target column with that model, and report M's gaze-target accuracy (T1.5).
Meetings given with --extra (no labels) are predicted with a model fitted on all labelled meetings.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..data.config import REPO_ROOT, load_config
from ..data.splits import series_of
from .calibrate import labelled_rows, to_samples
from .io import TrackRow, read_tracks, write_tracks
from .layout import layout_for_meeting
from .metrics import target_accuracy
from .vfoa import FEATURE_SETS, VfoaModel, fit, smooth


def relabel(rows: list[TrackRow], model: VfoaModel, smooth_window: int = 5) -> list[TrackRow]:
    """New gaze targets from `model`, smoothed per person in time order."""
    layout = layout_for_meeting(rows[0].meeting)
    out = list(rows)
    for person in {r.person for r in rows}:
        idx = sorted((i for i, r in enumerate(rows) if r.person == person), key=lambda i: rows[i].t)
        labs = [model.predict_row(layout.camera_of[person], rows[i]) for i in idx]
        for i, lab in zip(idx, smooth(labs, smooth_window)):
            out[i] = replace(rows[i], gaze_target=lab, target_off=None)
    return out


def nested_confusion(labelled: dict[str, list], tracks: Path, exclude_series: set[str] | frozenset = frozenset(),
                     features=FEATURE_SETS["eyes"], site: str = "idiap") -> dict:
    """Per-camera confusion matrices measured honestly: each training series is predicted by a model
    fitted on the *other* training series, so C reflects errors on unseen people. Series in
    `exclude_series` (the meeting under test) are never used. See quantify.py."""
    from .quantify import confusion_matrices

    series = sorted({series_of(m) for m in labelled} - set(exclude_series))
    triples = []
    for s in series:
        train = [p for m, rs in labelled.items() if series_of(m) not in set(exclude_series) | {s} for p in rs]
        if not train:
            continue
        layout = layout_for_meeting(next(iter(labelled)))
        model = fit(to_samples(train, layout, features), site, features=features)
        for m in (m for m in labelled if series_of(m) == s):
            pred = {(r.frame, r.person): r.gaze_target for r in relabel(read_tracks(tracks / f"{m}.csv"), model)}
            triples += [(layout.camera_of[r.person], t, pred[(r.frame, r.person)]) for r, t in labelled[m]]
    return confusion_matrices(triples)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="eyes")
    ap.add_argument("--extra", nargs="*", default=[], help="unlabelled meetings to predict with the all-data model")
    args = ap.parse_args(argv)

    feats = FEATURE_SETS[args.features]
    meetings = [m for m in cfg.meetings(args.meetings) if (args.tracks / f"{m}.csv").exists()]
    site = layout_for_meeting(meetings[0]).name
    labelled = {m: labelled_rows(args.data, m, args.tracks) for m in meetings}
    args.out.mkdir(parents=True, exist_ok=True)
    accs = []
    for held in meetings:
        # hold out the whole series: IS1001a/b/c are the same four people, so training on
        # IS1001a and testing on IS1001b would let the model see the test faces
        train = [p for m, rs in labelled.items() if series_of(m) != series_of(held) for p in rs]
        layout = layout_for_meeting(held)
        model = fit(to_samples(train, layout, feats), site, features=feats)
        rows = relabel(read_tracks(args.tracks / f"{held}.csv"), model)
        write_tracks(rows, args.out / f"{held}.csv")
        by_key = {(r.frame, r.person): r.gaze_target for r in rows}
        truth = [t for _, t in labelled[held]]
        pred = [by_key[(r.frame, r.person)] for r, _ in labelled[held]]
        majority = max(set(t for _, t in train), key=[t for _, t in train].count)
        acc = target_accuracy(truth, pred)
        accs.append(acc)
        print(f"{held}: gaze-target accuracy {acc:.1%} (majority '{majority}' {target_accuracy(truth, [majority] * len(truth)):.1%}), "
              f"{len(truth)} labelled frames, trained on {len(train)}")
    if accs:
        print(f"mean over {len(accs)} held-out meetings (series held out): {sum(accs) / len(accs):.1%} (T1.5 target 70%)")
    if args.extra:
        model = fit(to_samples([p for rs in labelled.values() for p in rs], layout_for_meeting(meetings[0]), feats),
                    site, features=feats)
        for m in args.extra:
            write_tracks(relabel(read_tracks(args.tracks / f"{m}.csv"), model), args.out / f"{m}.csv")
            print(f"{m}: predicted with the model fitted on all {len(meetings)} labelled meetings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
