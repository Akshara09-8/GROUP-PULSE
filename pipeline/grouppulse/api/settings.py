"""Backend settings from environment variables, with free-demo defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..data.config import REPO_ROOT


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # SQLite file locally; on Supabase use its Postgres URL (postgresql+psycopg://...)
    database_url: str = field(default_factory=lambda: _env(
        "GP_DATABASE_URL", f"sqlite:///{(REPO_ROOT / 'data' / 'grouppulse.db').as_posix()}"))
    files_dir: Path = field(default_factory=lambda: Path(_env("GP_FILES_DIR", str(REPO_ROOT / "data" / "uploads"))))
    max_upload_mb: int = field(default_factory=lambda: int(_env("GP_MAX_UPLOAD_MB", "200")))
    retention_hours: float = field(default_factory=lambda: float(_env("GP_RETENTION_HOURS", "24")))
    # free-tier guards (Supabase free: ~1 GB file storage): refuse new uploads beyond these
    max_active_uploads: int = field(default_factory=lambda: int(_env("GP_MAX_ACTIVE_UPLOADS", "50")))
    max_storage_mb: int = field(default_factory=lambda: int(_env("GP_MAX_STORAGE_MB", "900")))
    min_people: int = 2     # 3-5 is the designed range; 2 and 6 are accepted with a warning (T2.7)
    max_people: int = 6
