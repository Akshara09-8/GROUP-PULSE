"""Social metrics from a track table. Pure Python + NumPy; no video needed.

Definitions (plan doc, "Scoring formulas"):
- attention received by j = share of the other members' known-target gaze frames that land on j
- mutual gaze = share of pair-frames where i looks at j and j looks at i at the same time
- gaze following = share of gaze shifts to a target that another member copies within follow_s
- facing j = torso direction within facing_cone_deg of the direction to j's seat (needs layout)
- turned out = torso this far from the person's own median torso yaw (no layout needed)
- engagement per frame = (1 - body_weight) * gaze weight + body_weight * (not turned out)
- dominance = w_attention * attention received + w_facing * facing received (default: gaze only)
- rapport = mean of capped mutual gaze and gaze following (+ mean pairwise facing if enabled)
"""

from __future__ import annotations

import bisect
import re
import statistics
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..data.config import REPO_ROOT
from ..perception.geometry import angle_diff, direction_to
from ..perception.io import TrackRow
from ..perception.layout import SeatLayout

PERSON = re.compile(r"^P\d+$")
DEFAULT_SOCIAL = REPO_ROOT / "configs" / "social.toml"


@dataclass(frozen=True)
class SocialConfig:
    window_s: float = 10.0
    follow_s: float = 2.0
    gaze_weights: dict = field(default_factory=lambda: {
        "person": 1.0, "screen": 1.0, "whiteboard": 1.0, "table": 0.5, "away": 0.0, "unknown": 0.25})
    turned_out_deg: float = 45.0
    facing_cone_deg: float = 45.0
    body_weight: float = 0.3
    smooth_s: float = 5.0
    disengaged_below: float = 0.4
    disengaged_min_s: float = 30.0
    w_attention: float = 1.0
    w_facing: float = 0.0
    mutual_scale: float = 0.2
    rapport_use_facing: bool = False
    addressee_rule: str = "reengage"
    reengage_below: float = 0.4


def load_social_config(path: str | Path | None = None) -> SocialConfig:
    with Path(path or DEFAULT_SOCIAL).open("rb") as f:
        r = tomllib.load(f)
    return SocialConfig(
        window_s=r["windows"]["window_s"], follow_s=r["windows"]["follow_s"],
        gaze_weights=dict(r["gaze_weights"]),
        turned_out_deg=r["body"]["turned_out_deg"], facing_cone_deg=r["body"]["facing_cone_deg"],
        body_weight=r["body"]["body_weight"],
        smooth_s=r["disengagement"]["smooth_s"], disengaged_below=r["disengagement"]["below"],
        disengaged_min_s=r["disengagement"]["min_s"],
        w_attention=r["dominance"]["attention"], w_facing=r["dominance"]["facing"],
        mutual_scale=r["rapport"]["mutual_scale"],
        rapport_use_facing=r["rapport"].get("use_facing", False),
        addressee_rule=r["addressee"]["rule"], reengage_below=r["addressee"]["reengage_below"],
    )


# ---------------------------------------------------------------- data

@dataclass
class Series:
    """Track rows rearranged as aligned per-person time series."""
    people: list[str]
    t: list[float]                                   # time of each sampled frame
    target: dict[str, list[str]]                     # person -> gaze target per frame
    torso_yaw: dict[str, list[float | None]]
    torso_dir: dict[str, list[float | None]]

    @property
    def dt(self) -> float:
        steps = [b - a for a, b in zip(self.t, self.t[1:]) if b > a]
        return statistics.median(steps) if steps else 0.2

    @classmethod
    def from_rows(cls, rows: Iterable[TrackRow]) -> "Series":
        rows = list(rows)
        people = sorted({r.person for r in rows})
        frames = sorted({r.frame for r in rows})
        pos = {f: k for k, f in enumerate(frames)}
        t = [0.0] * len(frames)
        target = {p: ["unknown"] * len(frames) for p in people}
        yaw = {p: [None] * len(frames) for p in people}
        tdir = {p: [None] * len(frames) for p in people}
        for r in rows:
            k = pos[r.frame]
            t[k] = r.t
            target[r.person][k] = r.gaze_target or "unknown"
            yaw[r.person][k] = r.torso_yaw
            tdir[r.person][k] = r.torso_dir
        return cls(people, t, target, yaw, tdir)


def _known(target: str) -> bool:
    return target != "unknown"


def _weight(target: str, cfg: SocialConfig) -> float:
    return cfg.gaze_weights["person"] if PERSON.match(target) else cfg.gaze_weights.get(target, 0.0)


def _range(s: Series, idx: range | None) -> range:
    return idx if idx is not None else range(len(s.t))


# ---------------------------------------------------------------- gaze

def gaze_shares(s: Series, idx: range | None = None) -> dict[str, dict[str, float]]:
    """For each viewer, the share of their known-target frames spent on each target."""
    out = {}
    for p in s.people:
        tg = [s.target[p][k] for k in _range(s, idx) if _known(s.target[p][k])]
        counts: dict[str, int] = defaultdict(int)
        for x in tg:
            counts[x] += 1
        out[p] = {k: v / len(tg) for k, v in counts.items()} if tg else {}
    return out


def attention_received(s: Series, idx: range | None = None) -> dict[str, float]:
    """T2.1: share of all other members' known-target gaze frames that land on each person."""
    out = {}
    for j in s.people:
        hits = total = 0
        for i in s.people:
            if i == j:
                continue
            for k in _range(s, idx):
                x = s.target[i][k]
                if _known(x):
                    total += 1
                    hits += x == j
        out[j] = hits / total if total else 0.0
    return out


def mutual_gaze_rate(s: Series, idx: range | None = None) -> float:
    pairs = mutual = 0
    for a in range(len(s.people)):
        for b in range(a + 1, len(s.people)):
            i, j = s.people[a], s.people[b]
            for k in _range(s, idx):
                if _known(s.target[i][k]) and _known(s.target[j][k]):
                    pairs += 1
                    mutual += s.target[i][k] == j and s.target[j][k] == i
    return mutual / pairs if pairs else 0.0


def gaze_shifts(s: Series, p: str) -> list[tuple[float, str]]:
    """(time, new target) each time person p moves their gaze to a person or shared object."""
    out, prev = [], None
    for k, x in enumerate(s.target[p]):
        if x != prev and (PERSON.match(x) or x in ("screen", "whiteboard")):
            out.append((s.t[k], x))
        prev = x
    return out


def gaze_following_rate(s: Series, follow_s: float) -> float:
    """Share of gaze shifts that another member copies (shifts to the same target) within follow_s."""
    shifts = {p: gaze_shifts(s, p) for p in s.people}
    times = {p: [t for t, _ in sh] for p, sh in shifts.items()}
    events = followed = 0
    for i, sh in shifts.items():
        for t0, tgt in sh:
            events += 1
            for j, other in shifts.items():
                if j in (i, tgt):
                    continue
                a = bisect.bisect_right(times[j], t0)
                b = bisect.bisect_right(times[j], t0 + follow_s)
                if any(other[k][1] == tgt for k in range(a, b)):
                    followed += 1
                    break
    return followed / events if events else 0.0


# ---------------------------------------------------------------- body

def turned_out(s: Series, cfg: SocialConfig) -> dict[str, list[bool | None]]:
    """Torso far from the person's own median direction. Layout-free, so robust to bad geometry.

    Limitation: the median is the person's "usual" direction, so someone turned away for half the
    session or more shifts their own baseline and stops counting as turned out."""
    out = {}
    for p in s.people:
        ys = [y for y in s.torso_yaw[p] if y is not None]
        med = statistics.median(ys) if ys else None
        out[p] = [None if (y is None or med is None) else angle_diff(y, med) > cfg.turned_out_deg
                  for y in s.torso_yaw[p]]
    return out


def facing_received(s: Series, layout: SeatLayout | None, cfg: SocialConfig,
                    idx: range | None = None) -> dict[str, float]:
    """Share of other members' torso frames pointing at each person's seat (needs a layout)."""
    if layout is None:
        return {p: 0.0 for p in s.people}
    out = {}
    for j in s.people:
        hits = total = 0
        for i in s.people:
            if i == j or i not in layout.seats or j not in layout.seats:
                continue
            want = direction_to(layout.seats[i], layout.seats[j])
            for k in _range(s, idx):
                d = s.torso_dir[i][k]
                if d is not None:
                    total += 1
                    hits += angle_diff(d, want) <= cfg.facing_cone_deg
        out[j] = hits / total if total else 0.0
    return out


# ---------------------------------------------------------------- engagement

def engagement(s: Series, cfg: SocialConfig) -> dict[str, list[float]]:
    out_body = turned_out(s, cfg)
    out = {}
    for p in s.people:
        vals = []
        for k, x in enumerate(s.target[p]):
            g = _weight(x, cfg)
            b = out_body[p][k]
            vals.append(g if b is None else (1 - cfg.body_weight) * g + cfg.body_weight * (0.0 if b else 1.0))
        out[p] = vals
    return out


def _rolling_mean(xs: list[float], n: int) -> list[float]:
    if n <= 1:
        return list(xs)
    h, out, csum = n // 2, [], [0.0]
    for x in xs:
        csum.append(csum[-1] + x)
    for k in range(len(xs)):
        a, b = max(0, k - h), min(len(xs), k + h + 1)
        out.append((csum[b] - csum[a]) / (b - a))
    return out


def disengaged_spells(s: Series, eng: dict[str, list[float]], cfg: SocialConfig) -> list[dict]:
    """T2.3: runs where smoothed engagement stays below the threshold for at least min_s."""
    n = max(1, round(cfg.smooth_s / s.dt))
    spells = []
    for p in s.people:
        sm = _rolling_mean(eng[p], n)
        start = None
        for k, v in enumerate(sm + [1.0]):
            if v < cfg.disengaged_below and start is None:
                start = k
            elif v >= cfg.disengaged_below and start is not None:
                t0, t1 = s.t[start], s.t[k - 1] + s.dt
                if t1 - t0 >= cfg.disengaged_min_s:
                    spells.append({"person": p, "start": round(t0, 1), "end": round(t1, 1)})
                start = None
    return spells


# ---------------------------------------------------------------- scores

def dominance(att: dict[str, float], face: dict[str, float], cfg: SocialConfig) -> dict[str, float]:
    return {p: round(cfg.w_attention * att.get(p, 0) + cfg.w_facing * face.get(p, 0), 4) for p in att}


def rapport(mutual: float, following: float, facing_mean: float, cfg: SocialConfig) -> float:
    parts = [min(1.0, mutual / cfg.mutual_scale), following]
    if cfg.rapport_use_facing:
        parts.append(facing_mean)
    return round(sum(parts) / len(parts), 4)


def addressee(dom: dict[str, float], eng: dict[str, float], cfg: SocialConfig) -> dict:
    if not dom:
        return {"person": None, "reason": "no data"}
    least = min(eng, key=eng.get) if eng else None
    if cfg.addressee_rule == "reengage" and least is not None and eng[least] < cfg.reengage_below:
        return {"person": least, "reason": "re-engage least engaged"}
    return {"person": max(dom, key=dom.get), "reason": "address dominant"}


def adjusted_attention(s: Series, layout: SeatLayout, confusion: dict) -> dict[str, float]:
    """attention_received with each viewer's camera-specific gaze confusion undone (quantify.py).

    Same definition as attention_received (share of the others' known-target frames that land on j),
    but each viewer's target mix is first corrected from predicted to estimated-true."""
    from ..perception.quantify import adjusted_shares

    mass, total = defaultdict(float), defaultdict(float)
    for i in s.people:
        known = [x for x in s.target[i] if _known(x)]
        view = layout.camera_of.get(i)
        if not known or view not in confusion:
            shares = {k: v for k, v in gaze_shares(s)[i].items()}
        else:
            shares = adjusted_shares(known, confusion[view])
        for j in s.people:
            if j != i:
                mass[j] += len(known) * shares.get(j, 0.0)
                total[j] += len(known)
    return {j: (mass[j] / total[j] if total[j] else 0.0) for j in s.people}


def analyze(rows: Iterable[TrackRow], layout: SeatLayout | None = None,
            cfg: SocialConfig | None = None, confusion: dict | None = None) -> dict:
    """Session summary: per-person and group scores, windows timeline, spells, warnings.

    With `confusion` (per-camera matrices from the calibrated gaze model) and a layout, session-level
    attention received and dominance use adjusted counting; 10 s windows keep raw counts because a
    window holds too few frames to correct reliably."""
    cfg = cfg or SocialConfig()
    s = Series.from_rows(rows)
    warnings = []
    if not 3 <= len(s.people) <= 5:
        warnings.append(f"Group Pulse is designed for 3-5 people; this session has {len(s.people)}.")  # T2.7
    eng = engagement(s, cfg)
    raw_att = attention_received(s)
    use_adjusted = bool(confusion) and layout is not None
    att = adjusted_attention(s, layout, confusion) if use_adjusted else raw_att
    face = facing_received(s, layout, cfg)
    dom = dominance(att, face, cfg)
    mutual = mutual_gaze_rate(s)
    following = gaze_following_rate(s, cfg.follow_s)
    facing_mean = sum(face.values()) / len(face) if face else 0.0
    mean_eng = {p: round(sum(v) / len(v), 4) if v else 0.0 for p, v in eng.items()}
    spells = disengaged_spells(s, eng, cfg)

    per_window = max(1, round(cfg.window_s / s.dt))
    windows = []
    for a in range(0, len(s.t), per_window):
        idx = range(a, min(len(s.t), a + per_window))
        w_att = attention_received(s, idx)
        w_face = facing_received(s, layout, cfg, idx)
        w_dom = dominance(w_att, w_face, cfg)
        w_eng = {p: round(sum(eng[p][k] for k in idx) / len(idx), 4) for p in s.people}
        windows.append({
            "start": round(s.t[idx.start], 1), "end": round(s.t[idx.stop - 1] + s.dt, 1),
            "attention": {p: round(v, 4) for p, v in w_att.items()},
            "dominance": w_dom, "engagement": w_eng,
            "mutual_gaze": round(mutual_gaze_rate(s, idx), 4),
            "addressee": addressee(w_dom, w_eng, cfg),
        })

    ranked = sorted(dom, key=dom.get, reverse=True)
    return {
        "people": s.people,
        "duration_s": round(s.t[-1] + s.dt - s.t[0], 1) if s.t else 0.0,
        "warnings": warnings,
        "per_person": {p: {
            "attention_received": round(att[p], 4),
            "facing_received": round(face[p], 4),
            "dominance": dom[p],
            "engagement": mean_eng[p],
            "gaze_shares": {k: round(v, 4) for k, v in gaze_shares(s)[p].items()},
            "disengaged_s": round(sum(x["end"] - x["start"] for x in spells if x["person"] == p), 1),
        } for p in s.people},
        "group": {
            "dominant": ranked[0] if ranked else None,
            "dominance_rank": ranked,
            "mutual_gaze_rate": round(mutual, 4),
            "gaze_following_rate": round(following, 4),
            "facing_mean": round(facing_mean, 4),
            "rapport_index": rapport(mutual, following, facing_mean, cfg),
            "attention_method": "adjusted" if use_adjusted else "raw",
            "attention_raw": {p: round(v, 4) for p, v in raw_att.items()},
        },
        "disengaged_spells": spells,
        "windows": windows,
    }
