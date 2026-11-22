"""Export the demo sessions as static JSON for the server-free demo build (GitHub Pages).

    python scripts/export_demo.py            # writes web/public/demo/...

It loads demo/sessions/*.json into a throw-away in-memory API and saves the API's own responses,
so the static demo shows exactly what the real backend would:
    sessions.json, sessions/<id>.json, sessions/<id>/windows.json, sessions/<id>/people/<P>.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from fastapi.testclient import TestClient  # noqa: E402

from grouppulse.api import Settings, create_app  # noqa: E402
from grouppulse.api.import_ami import import_dir  # noqa: E402


def export(sessions_dir: Path, out: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(database_url=f"sqlite:///{(Path(tmp) / 'demo.db').as_posix()}", files_dir=Path(tmp) / "up")
        app = create_app(settings)
        ids = import_dir(app.state.store, sessions_dir)
        with TestClient(app) as c:
            def save(path: str, rel: str):
                r = c.get(path)
                r.raise_for_status()
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(r.json(), separators=(",", ":")), encoding="utf-8")

            save("/sessions", "sessions.json")
            for sid in ids:
                save(f"/sessions/{sid}", f"sessions/{sid}.json")
                save(f"/sessions/{sid}/windows", f"sessions/{sid}/windows.json")
                for label in c.get(f"/sessions/{sid}").json()["per_person"]:
                    save(f"/sessions/{sid}/people/{label}", f"sessions/{sid}/people/{label}.json")
        app.state.store.engine.dispose()   # Windows keeps the SQLite file locked until the pool closes
    return ids


if __name__ == "__main__":
    out = ROOT / "web" / "public" / "demo"
    ids = export(ROOT / "demo" / "sessions", out)
    print(f"exported {len(ids)} demo sessions to {out}: {' '.join(ids)}")
