"""Phase 5 evaluation: every experiment, one command, deterministic output.

Experiments
  E1  Gaze-target accuracy (T1.5). Idiap: each labelled meeting predicted by a model fitted only on
      other meeting series (different people). Edinburgh: ES2008a, one half predicts the other.
  E2  Dominance vs speaking time (T5.1 / T2.2): attention received from predicted gaze, from AMI's
      true gaze (upper bound), a seat baseline and the random expectation.
  E3  Ablation (T5.3): gaze only / orientation only / both, on meetings that have torso data.
  E4  Disengagement (T5.2) against a proxy: spells computed from AMI's true gaze labels, scored per
      10 s window where the labels cover the window.
  All means carry bootstrap 95% intervals over meetings (T5.5); fixed seeds make reruns identical (T5.4).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..data.annotations import find_ann_root
from ..data.config import load_config
from ..data.splits import series_of
from ..perception.calibrate import labelled_rows, to_samples
from ..perception.crossval import nested_confusion, relabel
from ..perception.io import read_tracks
from ..perception.layout import layout_for_meeting
from ..perception.metrics import target_accuracy
from ..perception.vfoa import FEATURE_SETS, fit
from ..social import SocialConfig, analyze
from ..social.report import seat_baseline, spearman, talk_rank, with_true_gaze
from .stats import bootstrap_ci, mean, prf, window_labels

FEATS = FEATURE_SETS["eyes"]
COVERAGE = 0.8      # share of a window's frames that must carry an AMI gaze label to score it (E4)


def _r(x, d=4):
    return None if x is None or x != x else round(float(x), d)


def _ci(items, stat):
    p, lo, hi = bootstrap_ci(items, stat)
    return {"value": _r(p), "ci95": [_r(lo), _r(hi)], "n": len(items)}


# ------------------------------------------------------------------ E1

def gaze_idiap(data: Path, tracks: Path, meetings: list[str]) -> tuple[list[dict], dict[str, list], dict[str, dict]]:
    """Per-meeting held-out gaze accuracy, the relabelled rows, and per-meeting confusion matrices
    (measured only on the other series, for adjusted counting)."""
    labelled = {m: labelled_rows(data, m, tracks) for m in meetings}
    site = layout_for_meeting(meetings[0]).name
    out, relabelled, confusions = [], {}, {}
    cache: dict[str, dict] = {}
    for held in meetings:
        s = series_of(held)
        if s not in cache:
            cache[s] = nested_confusion(labelled, tracks, {s}, FEATS, site)
        confusions[held] = cache[s]
        train = [p for m, rs in labelled.items() if series_of(m) != series_of(held) for p in rs]
        model = fit(to_samples(train, layout_for_meeting(held), FEATS), site, features=FEATS)
        rows = relabel(read_tracks(tracks / f"{held}.csv"), model)
        relabelled[held] = rows
        pred = {(r.frame, r.person): r.gaze_target for r in rows}
        truth = [t for _, t in labelled[held]]
        guess = [pred[(r.frame, r.person)] for r, _ in labelled[held]]
        majority = max(set(t for _, t in train), key=[t for _, t in train].count)
        out.append({"meeting": held, "accuracy": target_accuracy(truth, guess),
                    "majority_baseline": target_accuracy(truth, [majority] * len(truth)),
                    "labelled_frames": len(truth), "train_frames": len(train)})
    return out, relabelled, confusions


def gaze_edinburgh(data: Path, tracks: Path, meeting: str = "ES2008a") -> list[dict]:
    pairs = sorted(labelled_rows(data, meeting, tracks), key=lambda p: p[0].t)
    rows = read_tracks(tracks / f"{meeting}.csv")
    cut = pairs[len(pairs) // 2][0].t
    out = []
    for name, keep_train in (("first half -> second half", lambda t: t < cut), ("second half -> first half", lambda t: t >= cut)):
        train = [p for p in pairs if keep_train(p[0].t)]
        test_rows = [r for r in rows if not keep_train(r.t)]
        model = fit(to_samples(train, layout_for_meeting(meeting), FEATS), "edinburgh", features=FEATS)
        pred = {(r.frame, r.person): r.gaze_target for r in relabel(test_rows, model)}
        test = [p for p in pairs if not keep_train(p[0].t)]
        out.append({"meeting": meeting, "split": name,
                    "accuracy": target_accuracy([t for _, t in test], [pred[(r.frame, r.person)] for r, _ in test])})
    return out


# ------------------------------------------------------------------ E2 / E3

def rank_score(pred: list[str], truth: list[str]) -> dict:
    pred = [p for p in pred if p in truth]
    return {"top1": pred[0] == truth[0], "rho": spearman(pred, truth), "order": pred}


def dominance(meetings: dict[str, dict], truths: dict[str, list[str]], cfg: SocialConfig) -> list[dict]:
    """meetings: id -> {'pred': rows or None, 'true': rows or None}."""
    gaze = replace(cfg, w_attention=1.0, w_facing=0.0)
    out = []
    for m, src in meetings.items():
        row = {"meeting": m, "truth": truths[m], "seat_baseline": rank_score(seat_baseline(truths, m), truths[m])}
        for key in ("pred", "true"):
            if src.get(key):
                rank = analyze(src[key], layout_for_meeting(m), gaze)["group"]["dominance_rank"]
                row[f"gaze_{key}"] = rank_score(rank, truths[m])
        if src.get("pred") and src.get("confusion"):
            rank = analyze(src["pred"], layout_for_meeting(m), gaze, src["confusion"])["group"]["dominance_rank"]
            row["gaze_pred_adjusted"] = rank_score(rank, truths[m])
        out.append(row)
    return out


def ablation(meetings: dict[str, list], truths: dict[str, list[str]], cfg: SocialConfig) -> list[dict]:
    variants = {"gaze only": (1.0, 0.0), "orientation only": (0.0, 1.0), "both 0.6/0.4": (0.6, 0.4)}
    out = []
    for m, rows in meetings.items():
        row = {"meeting": m}
        for name, (wa, wf) in variants.items():
            rank = analyze(rows, layout_for_meeting(m), replace(cfg, w_attention=wa, w_facing=wf))["group"]["dominance_rank"]
            row[name] = rank_score(rank, truths[m])
        out.append(row)
    return out


# ------------------------------------------------------------------ E4

def coverage(true_rows, people: list[str], window_s: float) -> dict[tuple[int, str], float]:
    tot, lab = {}, {}
    for r in true_rows:
        k = (int(r.t // window_s), r.person)
        tot[k] = tot.get(k, 0) + 1
        lab[k] = lab.get(k, 0) + (r.gaze_target != "unknown")
    return {k: lab[k] / tot[k] for k in tot}


def disengagement(pairs: dict[str, tuple[list, list]], cfg: SocialConfig) -> list[dict]:
    out = []
    for m, (pred_rows, true_rows) in pairs.items():
        p = analyze(pred_rows, layout_for_meeting(m), cfg)
        t = analyze(true_rows, layout_for_meeting(m), cfg)
        people, end = t["people"], t["duration_s"]
        cov = coverage(true_rows, people, cfg.window_s)
        keep = {k for k, c in cov.items() if c >= COVERAGE}
        tl = {k: v for k, v in window_labels(t["disengaged_spells"], people, end, cfg.window_s).items() if k in keep}
        pl = {k: v for k, v in window_labels(p["disengaged_spells"], people, end, cfg.window_s).items() if k in keep}
        # continuous agreement: window engagement from predicted vs true gaze (steadier than rare-event F1)
        te = {(i, q): w["engagement"][q] for i, w in enumerate(t["windows"]) for q in people}
        pe = {(i, q): w["engagement"][q] for i, w in enumerate(p["windows"]) for q in people}
        shared = sorted(k for k in keep if k in te and k in pe)
        out.append({"meeting": m, "truth": tl, "pred": pl, "scored_windows": len(tl),
                    "true_positive_windows": sum(tl.values()), **prf(tl, pl),
                    "engagement_r": pearson([pe[k] for k in shared], [te[k] for k in shared])})
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def pooled_f1(items: list[dict]) -> float:
    truth, pred = {}, {}
    for it in items:
        truth.update({(it["meeting"],) + k: v for k, v in it["truth"].items()})
        pred.update({(it["meeting"],) + k: v for k, v in it["pred"].items()})
    return prf(truth, pred)["f1"]


# ------------------------------------------------------------------ run

def run(data: Path, idiap_tracks: Path | None, edinburgh_tracks: Path | None) -> dict:
    cfg_all = load_config()
    social = SocialConfig()
    ann = find_ann_root(data / "annotations")
    res: dict = {"settings": {"gaze_features": list(FEATS), "bootstrap": "2000 resamples of meetings, seed 13",
                              "window_s": social.window_s, "coverage_for_E4": COVERAGE}}

    idiap = [m for m in cfg_all.meetings("focus") if m.startswith("IS") and idiap_tracks and (idiap_tracks / f"{m}.csv").exists()]
    relabelled, confusions = {}, {}
    if len(idiap) >= 3:
        e1, relabelled, confusions = gaze_idiap(data, idiap_tracks, idiap)
        res["E1_gaze_idiap"] = {"per_meeting": [{k: _r(v) if isinstance(v, float) else v for k, v in x.items()} for x in e1],
                                "mean_accuracy": _ci([x["accuracy"] for x in e1], mean),
                                "mean_majority_baseline": _ci([x["majority_baseline"] for x in e1], mean),
                                "series": sorted({series_of(m) for m in idiap})}
    if edinburgh_tracks and (edinburgh_tracks / "ES2008a.csv").exists():
        e1e = gaze_edinburgh(data, edinburgh_tracks)
        res["E1_gaze_edinburgh"] = {"per_split": [{**x, "accuracy": _r(x["accuracy"])} for x in e1e],
                                    "mean_accuracy": _r(mean([x["accuracy"] for x in e1e]))}

    # E2: dominance, per site so seat baselines stay within one room layout
    for site, group in (("idiap", idiap), ("edinburgh", ["ES2002a", "ES2002b", "ES2003a", "ES2008a"])):
        meetings = {}
        for m in group:
            if site == "idiap":
                orig = read_tracks(idiap_tracks / f"{m}.csv")
                meetings[m] = {"pred": relabelled.get(m), "true": with_true_gaze(orig, ann, m),
                               "confusion": confusions.get(m)}
            elif edinburgh_tracks and (edinburgh_tracks / f"{m}.csv").exists():
                rows = read_tracks(edinburgh_tracks / f"{m}.csv")
                # ES2008a trained the Edinburgh gaze model: its predicted gaze is in-sample, so true gaze only
                meetings[m] = {"pred": None if m == "ES2008a" else rows,
                               "true": with_true_gaze(rows, ann, m) if m == "ES2008a" else None}
        if len(meetings) < 2:
            continue
        truths = {m: talk_rank(ann, m) for m in meetings}
        e2 = dominance(meetings, truths, social)
        summary = {}
        for key in ("gaze_pred", "gaze_pred_adjusted", "gaze_true", "seat_baseline"):
            items = [x[key] for x in e2 if key in x]
            if items:
                summary[key] = {"top1": _ci([float(i["top1"]) for i in items], mean),
                                "mean_rank_corr": _ci([i["rho"] for i in items], mean)}
        n_people = {len(t) for t in truths.values()}
        summary["random"] = {"top1": round(1 / min(n_people), 4), "mean_rank_corr": 0.0}
        res[f"E2_dominance_{site}"] = {"per_meeting": [_clean(x) for x in e2], "summary": summary}

    # E3: ablation where torso data exists (Edinburgh close-up torso)
    if edinburgh_tracks:
        abl = {m: read_tracks(edinburgh_tracks / f"{m}.csv") for m in ("ES2002a", "ES2002b", "ES2003a")
               if (edinburgh_tracks / f"{m}.csv").exists()}
        abl = {m: rows for m, rows in abl.items() if any(r.torso_dir is not None for r in rows)}
        if abl:
            truths = {m: talk_rank(ann, m) for m in abl}
            e3 = ablation(abl, truths, social)
            res["E3_ablation_edinburgh"] = {
                "per_meeting": [_clean(x) for x in e3],
                "summary": {v: {"top1": _ci([float(x[v]["top1"]) for x in e3], mean),
                                "mean_rank_corr": _ci([x[v]["rho"] for x in e3], mean)}
                            for v in ("gaze only", "orientation only", "both 0.6/0.4")},
                "note": "orientation = facing received from the uncalibrated seat layout"}

    # E4: disengagement vs gaze-label proxy (Idiap, held-out gaze)
    if relabelled:
        pairs = {m: (relabelled[m], with_true_gaze(read_tracks(idiap_tracks / f"{m}.csv"), ann, m)) for m in idiap}
        e4 = disengagement(pairs, social)
        res["E4_disengagement_idiap"] = {
            "per_meeting": [{k: (_r(v) if isinstance(v, float) else v) for k, v in x.items() if k not in ("truth", "pred")}
                            for x in e4],
            "pooled_f1": _ci(e4, pooled_f1),
            "mean_engagement_r": _ci([x["engagement_r"] for x in e4 if x["engagement_r"] == x["engagement_r"]], mean),
            "note": "ground truth = spells computed from AMI's true gaze labels (a proxy, not observed disengagement)"}
    return res


def _clean(x: dict) -> dict:
    out = {}
    for k, v in x.items():
        if isinstance(v, dict):
            out[k] = {kk: (_r(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def dumps(res: dict) -> str:
    return json.dumps(res, indent=2, sort_keys=True)
