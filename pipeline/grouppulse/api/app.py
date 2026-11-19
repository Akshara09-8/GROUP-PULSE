"""HTTP API. Interactive docs at /docs (test T3.7).

Endpoints
  GET    /health
  POST   /sessions                    upload a video (multipart) -> 202 {id, status}
  GET    /sessions                    list sessions, newest first
  GET    /sessions/{id}               status; scores when done
  GET    /sessions/{id}/windows       10 s timeline (attention, dominance, engagement, addressee)
  GET    /sessions/{id}/people/{P}    one participant: scores, engagement over time, spells
  DELETE /sessions/{id}               remove a session, its results and its video
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .files import LocalFiles, NotVideo, TooLarge
from .settings import Settings
from .store import Store
from .worker import Processor, Worker, fail_orphans, single_camera_processor

DEMO_FULL = "The demo is full right now. Uploads are deleted after 24 hours, so please try again later."


def purge_expired(store: Store, files: LocalFiles) -> int:
    """Retention (test T6.3): delete uploaded videos past expiry; the derived numbers stay."""
    n = 0
    for sid, path in store.expired_videos():
        files.delete(path)
        store.clear_video(sid)
        n += 1
    return n


def create_app(settings: Settings | None = None, store: Store | None = None,
               files: LocalFiles | None = None, processor: Processor | None = None) -> FastAPI:
    settings = settings or Settings()
    store = store or Store(settings.database_url)
    files = files or LocalFiles(settings.files_dir)
    worker = Worker(store, processor or single_camera_processor)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        fail_orphans(store)
        purge_expired(store, files)
        yield
        worker.shutdown()

    app = FastAPI(title="Group Pulse API", version="0.1.0", lifespan=lifespan,
                  description="Who leads and who disengages, from gaze and body orientation.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.store, app.state.files, app.state.worker = store, files, worker

    @app.get("/health")
    def health():
        n, b = store.active_uploads()
        return {"ok": True, "active_uploads": n, "upload_mb": round(b / (1 << 20), 1)}

    @app.post("/sessions", status_code=202)
    def upload(file: UploadFile = File(...), title: str = Form("Untitled session"),
               people: int = Form(...), consent: bool = Form(False)):
        if not consent:
            raise HTTPException(400, "Please confirm that everyone in the video agreed to be analysed.")
        if not settings.min_people <= people <= settings.max_people:
            raise HTTPException(422, f"People in view must be between {settings.min_people} and "
                                     f"{settings.max_people} (Group Pulse is designed for 3-5).")
        purge_expired(store, files)
        n, used = store.active_uploads()
        if n >= settings.max_active_uploads or used >= settings.max_storage_mb * (1 << 20):
            raise HTTPException(507, DEMO_FULL)                                    # T3.6
        sid = uuid.uuid4().hex[:12]
        try:
            path, size = files.save(file.file, sid, Path(file.filename or "").suffix,
                                    settings.max_upload_mb * (1 << 20))
        except NotVideo as e:
            raise HTTPException(415, f"That file can't be used: {e}. Upload an MP4, MOV, AVI or WebM video.")
        except TooLarge as e:
            raise HTTPException(413, f"That file is too large: {e}.")
        if used + size > settings.max_storage_mb * (1 << 20):
            files.delete(path)
            raise HTTPException(507, DEMO_FULL)
        store.create_session(title.strip()[:200] or "Untitled session", "upload", people, path, size,
                             settings.retention_hours, session_id=sid)
        worker.submit(sid, path, people)
        return {"id": sid, "status": "queued"}

    @app.get("/sessions")
    def list_sessions():
        return store.list_sessions()

    def _get(sid: str) -> dict:
        s = store.get_session(sid)
        if s is None:
            raise HTTPException(404, "No session with that id.")
        return s

    @app.get("/sessions/{sid}")
    def get_session(sid: str):
        return _get(sid)

    @app.get("/sessions/{sid}/windows")
    def get_windows(sid: str):
        s = _get(sid)
        if s["status"] != "done":
            raise HTTPException(409, f"Session is {s['status']}; the timeline is ready when it is done.")
        return store.get_windows(sid)

    @app.get("/sessions/{sid}/people/{label}")
    def get_person(sid: str, label: str):
        s = _get(sid)
        if s["status"] != "done" or label not in (s.get("per_person") or {}):
            raise HTTPException(404, f"No results for {label} in this session.")
        wins = store.get_windows(sid)
        return {
            "label": label,
            **s["per_person"][label],
            "group_mean_engagement": round(sum(p["engagement"] for p in s["per_person"].values())
                                           / len(s["per_person"]), 4),
            "engagement_over_time": [{"start": w["start"], "end": w["end"], "value": w["engagement"].get(label)}
                                     for w in wins],
            "disengaged_spells": [x for x in s["disengaged_spells"] if x["person"] == label],
        }

    @app.delete("/sessions/{sid}", status_code=204)
    def delete_session(sid: str):
        path = store.delete_session(sid)
        if path is None:
            raise HTTPException(404, "No session with that id.")
        files.delete(path)

    return app
