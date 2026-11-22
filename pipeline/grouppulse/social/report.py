"""Turn a track table into a session summary (JSON) and score dominance against AMI.

    # one session -> JSON for the API / UI
    python -m grouppulse.social.report --tracks data/tracks/ES2002a.csv --out data/sessions/ES2002a.json

    # T2.2: dominance ranking vs who spoke most, over several meetings
    python -m grouppulse.social.report --data D --tracks-dir D/tracks_full --evaluate ES2002a ES2002b ES2003a

    # same, but with AMI's true gaze labels instead of predicted gaze (only `focus` meetings)
    python -m grouppulse.social.report --data D --tracks-dir D/tracks_full --evaluate ES2008a --true-gaze

Ground truth for dominance is AMI speaking time (the person who talks most). AMI has no dominance
ratings; ELEA would provide them later. Because the research question is about gaze and body only,
speaking time is used for scoring, never as an input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..data.annotations import find_ann_root, load_focus, load_segments, load_speakers, seat_of_speaker, speaking_time
from ..data.config import REPO_ROOT
from ..perception.evaluate import truth_at
from ..perception.io import TrackRow, read_tracks
from ..perception.layout import layout_for_meeting
from .metrics import SocialConfig, analyze, load_social_config


def site_confusion(layout) -> dict:
    """Confusion matrices saved with the room's calibrated gaze model, if any (enables adjusted counting)."""
    from ..perception.vfoa import VfoaModel, default_path

    if layout is None or not default_path(layout.name).exists():
        return {}
    return VfoaModel.load(default_path(layout.name)).confusion


def spearman(a: list[str], b: list[str]) -> float:
    """Rank correlation of two orderings of the same items (no ties)."""
    n = len(a)
    if n < 2:
        return float("nan")
    rb = {x: k for k, x in enumerate(b)}
    d2 = sum((k - rb[x]) ** 2 for k, x in enumerate(a))
    return 1 - 6 * d2 / (n * (n * n - 1))


def talk_rank(ann: Path, meeting: str) -> list[str]:
    seat = seat_of_speaker(load_speakers(ann, meeting))
    st = speaking_time(load_segments(ann, meeting))
    return [seat[k] for k in sorted(st, key=st.get, reverse=True) if k in seat]


def with_true_gaze(rows: list[TrackRow], ann: Path, meeting: str) -> list[TrackRow]:
    """Replace predicted gaze targets with AMI focus labels (rows outside labelled spans -> unknown)."""
    seat = seat_of_speaker(load_speakers(ann, meeting))
    spans: dict[str, list] = {}
    for f in load_focus(ann, meeting):
        if f.speaker in seat:
            spans.setdefault(seat[f.speaker], []).append(f)
    out = []
    for r in rows:
        sp = truth_at(spans.get(r.person, []), r.t)
        out.append(replace(r, gaze_target=seat.get(sp.target, sp.target) if sp else "unknown"))
    return out


def seat_baseline(truths: dict[str, list[str]], held: str) -> list[str]:
    """Rank seats by how talkative their occupants were in the *other* meetings (mean rank).

    AMI seating is not random (the project manager tends to take the same seat), so any
    gaze-based ranking has to beat this to show it measures something beyond seating."""
    seats = sorted({s for m, t in truths.items() if m != held for s in t} | set(truths[held]))
    def mean_rank(s):
        ranks = [t.index(s) for m, t in truths.items() if m != held and s in t]
        return sum(ranks) / len(ranks) if ranks else 99
    return sorted((s for s in seats if s in truths[held]), key=lambda s: (mean_rank(s), s))


def evaluate(data: Path, tracks_dir: Path, meetings: list[str], cfg: SocialConfig, true_gaze: bool) -> int:
    ann = find_ann_root(data / "annotations")
    tracks = {m: read_tracks(tracks_dir / f"{m}.csv") for m in meetings}
    has_torso = all(any(r.torso_dir is not None for r in rows) for rows in tracks.values())
    variants = {"gaze only": replace(cfg, w_attention=1.0, w_facing=0.0)}
    if has_torso:   # without torso data every facing score is 0 and the "ranking" is just seat order
        variants["facing only"] = replace(cfg, w_attention=0.0, w_facing=1.0)
        variants["plan mix 0.6/0.4"] = replace(cfg, w_attention=0.6, w_facing=0.4)
    truths = {m: talk_rank(ann, m) for m in meetings}
    names = [*variants, "seat baseline"]
    hits = {k: 0 for k in names}
    rhos = {k: [] for k in names}
    for m in meetings:
        rows = with_true_gaze(tracks[m], ann, m) if true_gaze else tracks[m]
        truth = truths[m]
        line = [f"{m}: most talkative {truth[0]} (order {' > '.join(truth)})"]
        preds = {name: [p for p in analyze(rows, layout_for_meeting(m), c)["group"]["dominance_rank"] if p in truth]
                 for name, c in variants.items()}
        preds["seat baseline"] = seat_baseline(truths, m)
        for name, pred in preds.items():
            hit = pred[0] == truth[0]
            hits[name] += hit
            rho = spearman(pred, truth)
            rhos[name].append(rho)
            line.append(f"    {name:<16} top {pred[0]} {'hit ' if hit else 'miss'}  rank corr {rho:+.2f}")
        print("\n".join(line))
    n = len(meetings)
    print(f"\nover {n} meetings (chance top-1 = 25%{'' if has_torso else '; no torso data, facing skipped'}):")
    for name in names:
        print(f"  {name:<16} top-1 {hits[name]}/{n} = {hits[name] / n:.0%}   mean rank corr {sum(rhos[name]) / n:+.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, help="one track table to summarise")
    ap.add_argument("--out", type=Path, help="where to write the JSON summary")
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--tracks-dir", type=Path, default=REPO_ROOT / "data" / "tracks")
    ap.add_argument("--evaluate", nargs="+", metavar="MEETING", help="score dominance on these meetings")
    ap.add_argument("--true-gaze", action="store_true", help="use AMI focus labels instead of predicted gaze")
    ap.add_argument("--window", type=float, default=None, help="timeline window in seconds")
    args = ap.parse_args(argv)

    cfg = load_social_config()
    if args.window:
        cfg = replace(cfg, window_s=args.window)
    if args.evaluate:
        return evaluate(args.data, args.tracks_dir, args.evaluate, cfg, args.true_gaze)
    if not args.tracks:
        ap.error("give --tracks or --evaluate")
    rows = read_tracks(args.tracks)
    meeting = rows[0].meeting if rows else ""
    layout = layout_for_meeting(meeting) if meeting else None
    summ = analyze(rows, layout, cfg, site_confusion(layout))
    summ = {"session": meeting, **summ}
    text = json.dumps(summ, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        g = summ["group"]
        print(f"{meeting}: dominant {g['dominant']}, rapport {g['rapport_index']}, "
              f"{len(summ['disengaged_spells'])} disengaged spells, {len(summ['windows'])} windows -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
