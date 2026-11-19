# API (Phase 3)

The backend code lives in `pipeline/grouppulse/api/` so it installs with the rest of the package.

```bash
pip install -e ".[api]"
python -m grouppulse.api.import_ami path/to/ami-data/sessions   # load AMI demo sessions
python -m grouppulse.api --port 8000                                    # docs at http://127.0.0.1:8000/docs
```

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Liveness and current upload storage use |
| `POST /sessions` | Multipart upload: `file`, `people` (2–6), `consent=true`, `title`. Returns 202 `{id, status}` |
| `GET /sessions` | All sessions, newest first, with dominant person and rapport |
| `GET /sessions/{id}` | Status; when done, per-person scores, group scores, disengaged spells, warnings |
| `GET /sessions/{id}/windows` | 10 s timeline: attention, dominance, engagement per person; mutual gaze; robot addressee |
| `GET /sessions/{id}/people/{P}` | One person: scores, engagement over time, their spells, group mean |
| `DELETE /sessions/{id}` | Removes results and the video |

Upload checks: consent required (400), 2–6 people (422), real video content checked from the file's first bytes (415), size limit (413), free-tier quota (507 "demo is full").

## Settings (environment variables)

| Variable | Default | Notes |
| --- | --- | --- |
| `GP_DATABASE_URL` | `sqlite:///data/grouppulse.db` | Supabase: `postgresql+psycopg://...` from the project's connection settings; `pip install -e ".[supabase]"` |
| `GP_FILES_DIR` | `data/uploads` | Uploaded videos |
| `GP_RETENTION_HOURS` | `24` | Videos are deleted after this; derived numbers are kept |
| `GP_MAX_UPLOAD_MB` | `200` | Per file |
| `GP_MAX_ACTIVE_UPLOADS` / `GP_MAX_STORAGE_MB` | `50` / `900` | Free-tier guard (Supabase free storage is about 1 GB) |

## Not done yet

- **Uploads are accepted but fail with a clear message.** Phase 1 reads one close-up camera per person (the AMI setup); an upload is one camera filming the whole group, which needs a single-camera reader (multi-face tracking, image-plane gaze targets).
- **Supabase is untested.** The same SQLAlchemy code runs on Postgres, but it has only been run on SQLite here. Uploaded files are stored on local disk; Supabase Storage would need its own small adapter.
- The job queue is in-process (FastAPI single worker thread). A restart marks unfinished jobs as failed.
