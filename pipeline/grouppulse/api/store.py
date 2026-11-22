"""Database layer (SQLAlchemy Core). Same SQL on SQLite (local) and Postgres (Supabase).

Tables follow the plan doc: sessions, participant_metrics, group_metrics, windows (one row per
participant per window), window_group (group values and robot addressee per window), spells.
Only derived numbers are stored; raw video lives in files.py and is deleted after retention.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, create_engine, delete, func, insert, select, update)

meta = MetaData()

sessions = Table(
    "sessions", meta,
    Column("id", String(32), primary_key=True),
    Column("title", String(200), nullable=False),
    Column("source", String(20), nullable=False),          # "upload" or "ami"
    Column("status", String(20), nullable=False),          # queued, processing, done, failed
    Column("error", String(500)),
    Column("people", Integer),
    Column("duration_s", Float),
    Column("video_path", String(500)),                     # null once deleted by retention
    Column("video_bytes", Integer, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True)),         # when the uploaded video is deleted
)

participant_metrics = Table(
    "participant_metrics", meta,
    Column("session_id", String(32), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("label", String(8), primary_key=True),
    Column("attention_received", Float), Column("facing_received", Float),
    Column("dominance", Float), Column("engagement", Float), Column("disengaged_s", Float),
    Column("gaze_shares", JSON),
)

group_metrics = Table(
    "group_metrics", meta,
    Column("session_id", String(32), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("dominant", String(8)), Column("dominance_rank", JSON),
    Column("rapport_index", Float), Column("mutual_gaze_rate", Float),
    Column("gaze_following_rate", Float), Column("facing_mean", Float),
    Column("warnings", JSON),
)

windows = Table(
    "windows", meta,
    Column("session_id", String(32), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("label", String(8), primary_key=True),
    Column("t_start", Float), Column("t_end", Float),
    Column("attention", Float), Column("dominance", Float), Column("engagement", Float),
)

window_group = Table(
    "window_group", meta,
    Column("session_id", String(32), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("t_start", Float), Column("t_end", Float),
    Column("mutual_gaze", Float), Column("addressee", String(8)), Column("addressee_reason", String(80)),
)

spells = Table(
    "spells", meta,
    Column("session_id", String(32), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("label", String(8), primary_key=True),
    Column("t_start", Float, primary_key=True), Column("t_end", Float),
)

CHILD_TABLES = (participant_metrics, group_metrics, windows, window_group, spells)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(d: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat them as UTC."""
    return d.replace(tzinfo=timezone.utc) if d is not None and d.tzinfo is None else d


class Store:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            # SQLite cannot create a missing folder (fresh clone: data/ is git-ignored)
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, **kw)
        meta.create_all(self.engine)

    # ---- sessions
    def create_session(self, title: str, source: str, people: int | None = None,
                       video_path: str | None = None, video_bytes: int = 0,
                       retention_hours: float | None = None, session_id: str | None = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        created = now()
        with self.engine.begin() as c:
            c.execute(insert(sessions).values(
                id=sid, title=title, source=source, status="queued" if source == "upload" else "done",
                people=people, video_path=video_path, video_bytes=video_bytes, created_at=created,
                expires_at=created + timedelta(hours=retention_hours) if retention_hours and video_path else None))
        return sid

    def set_status(self, sid: str, status: str, error: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(update(sessions).where(sessions.c.id == sid).values(status=status, error=error))

    def save_summary(self, sid: str, summ: dict) -> None:
        """Store an analyze() summary and mark the session done (replaces earlier results)."""
        with self.engine.begin() as c:
            for t in CHILD_TABLES:
                c.execute(delete(t).where(t.c.session_id == sid))
            c.execute(insert(participant_metrics), [
                {"session_id": sid, "label": p, **{k: v for k, v in d.items()}}
                for p, d in summ["per_person"].items()])
            g = summ["group"]
            c.execute(insert(group_metrics).values(
                session_id=sid, dominant=g["dominant"], dominance_rank=g["dominance_rank"],
                rapport_index=g["rapport_index"], mutual_gaze_rate=g["mutual_gaze_rate"],
                gaze_following_rate=g["gaze_following_rate"], facing_mean=g["facing_mean"],
                warnings=summ.get("warnings", [])))
            wrows, grows = [], []
            for i, w in enumerate(summ["windows"]):
                for p in summ["people"]:
                    wrows.append({"session_id": sid, "idx": i, "label": p, "t_start": w["start"], "t_end": w["end"],
                                  "attention": w["attention"].get(p), "dominance": w["dominance"].get(p),
                                  "engagement": w["engagement"].get(p)})
                grows.append({"session_id": sid, "idx": i, "t_start": w["start"], "t_end": w["end"],
                              "mutual_gaze": w["mutual_gaze"], "addressee": w["addressee"]["person"],
                              "addressee_reason": w["addressee"]["reason"]})
            if wrows:
                c.execute(insert(windows), wrows)
                c.execute(insert(window_group), grows)
            if summ["disengaged_spells"]:
                c.execute(insert(spells), [{"session_id": sid, "label": s["person"], "t_start": s["start"],
                                            "t_end": s["end"]} for s in summ["disengaged_spells"]])
            c.execute(update(sessions).where(sessions.c.id == sid).values(
                status="done", error=None, people=len(summ["people"]), duration_s=summ["duration_s"]))

    def _session_row(self, c, sid: str):
        return c.execute(select(sessions).where(sessions.c.id == sid)).mappings().first()

    def list_sessions(self) -> list[dict]:
        with self.engine.connect() as c:
            q = (select(sessions, group_metrics.c.dominant, group_metrics.c.rapport_index)
                 .join(group_metrics, group_metrics.c.session_id == sessions.c.id, isouter=True)
                 .order_by(sessions.c.created_at.desc()))
            return [self._public(r) for r in c.execute(q).mappings()]

    def get_session(self, sid: str) -> dict | None:
        with self.engine.connect() as c:
            s = self._session_row(c, sid)
            if s is None:
                return None
            out = self._public(s)
            if s["status"] == "done":
                g = c.execute(select(group_metrics).where(group_metrics.c.session_id == sid)).mappings().first()
                out["group"] = {k: v for k, v in dict(g).items() if k not in ("session_id", "warnings")} if g else None
                out["warnings"] = (g["warnings"] if g else []) or []
                out["per_person"] = {r["label"]: {k: v for k, v in dict(r).items() if k not in ("session_id", "label")}
                                     for r in c.execute(select(participant_metrics)
                                                        .where(participant_metrics.c.session_id == sid)
                                                        .order_by(participant_metrics.c.label)).mappings()}
                out["disengaged_spells"] = [{"person": r["label"], "start": r["t_start"], "end": r["t_end"]}
                                            for r in c.execute(select(spells).where(spells.c.session_id == sid)
                                                               .order_by(spells.c.t_start)).mappings()]
            return out

    def get_windows(self, sid: str) -> list[dict]:
        with self.engine.connect() as c:
            out = {r["idx"]: {"start": r["t_start"], "end": r["t_end"], "mutual_gaze": r["mutual_gaze"],
                              "addressee": {"person": r["addressee"], "reason": r["addressee_reason"]},
                              "attention": {}, "dominance": {}, "engagement": {}}
                   for r in c.execute(select(window_group).where(window_group.c.session_id == sid)).mappings()}
            for r in c.execute(select(windows).where(windows.c.session_id == sid)).mappings():
                w = out[r["idx"]]
                w["attention"][r["label"]] = r["attention"]
                w["dominance"][r["label"]] = r["dominance"]
                w["engagement"][r["label"]] = r["engagement"]
            return [out[i] for i in sorted(out)]

    def video_path(self, sid: str) -> str | None:
        with self.engine.connect() as c:
            s = self._session_row(c, sid)
        return s["video_path"] if s is not None else None

    def delete_session(self, sid: str) -> str | None:
        """Remove a session and its results; returns its video path (for the caller to delete)."""
        with self.engine.begin() as c:
            s = self._session_row(c, sid)
            if s is None:
                return None
            for t in CHILD_TABLES:
                c.execute(delete(t).where(t.c.session_id == sid))
            c.execute(delete(sessions).where(sessions.c.id == sid))
            return s["video_path"] or ""

    # ---- retention and free-tier guards
    def expired_videos(self, at: datetime | None = None) -> list[tuple[str, str]]:
        at = at or now()
        with self.engine.connect() as c:
            rows = c.execute(select(sessions.c.id, sessions.c.video_path, sessions.c.expires_at)
                             .where(sessions.c.video_path.is_not(None))).all()
        return [(sid, path) for sid, path, exp in rows if exp is not None and _aware(exp) <= at]

    def clear_video(self, sid: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(sessions).where(sessions.c.id == sid).values(video_path=None, video_bytes=0))

    def active_uploads(self) -> tuple[int, int]:
        """(number of sessions still holding a video, total bytes of those videos)."""
        with self.engine.connect() as c:
            n, b = c.execute(select(func.count(), func.coalesce(func.sum(sessions.c.video_bytes), 0))
                             .where(sessions.c.video_path.is_not(None))).one()
        return int(n), int(b)

    @staticmethod
    def _public(r) -> dict:
        d = dict(r)
        d.pop("video_path", None)
        d["has_video"] = bool(r.get("video_path"))
        for k in ("created_at", "expires_at"):
            v = _aware(d.get(k))
            d[k] = v.isoformat() if v else None
        return d
