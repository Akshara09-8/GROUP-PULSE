"""Phase 3 backend: FastAPI app, storage, background worker, retention.

    python -m grouppulse.api                 # serve on http://127.0.0.1:8000 (docs at /docs)
    python -m grouppulse.api.import_ami D/sessions   # load pre-processed AMI demo sessions

Configuration comes from environment variables (see settings.py). The same code runs on a local
SQLite file and on Supabase Postgres: set DATABASE_URL to Supabase's connection string.
"""

from .app import create_app
from .settings import Settings

__all__ = ["create_app", "Settings"]
