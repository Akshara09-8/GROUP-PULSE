"""Phase 6 launch: retention loop (T6.3), error tracking (T6.4), demo data (T6.2), deploy config."""

import io
import json
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from grouppulse.api import Settings, create_app  # noqa: E402
from grouppulse.api.app import init_error_tracking  # noqa: E402
from grouppulse.api.import_ami import import_dir  # noqa: E402
from grouppulse.api.store import Store  # noqa: E402
from grouppulse.perception.io import TrackRow  # noqa: E402
from grouppulse.social import analyze  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024


def tiny_summary(video, people):
    rows = [TrackRow("u", k, k / 5, p, True, 0.0, 0.0, None, "P1" if p != "P1" else "P2", None, None, 0.0, False)
            for k in range(30) for p in ("P1", "P2", "P3")]
    return analyze(rows, None)


def test_t63_retention_loop_deletes_expired_video_without_new_uploads(tmp_path):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'g.db').as_posix()}", files_dir=tmp_path / "up",
                 retention_hours=0.3 / 3600, purge_every_s=0.2)          # expire after 0.3 s, sweep every 0.2 s
    with TestClient(create_app(s, processor=tiny_summary)) as c:
        sid = c.post("/sessions", files={"file": ("m.mp4", io.BytesIO(MP4), "video/mp4")},
                     data={"people": "3", "consent": "true"}).json()["id"]
        for _ in range(100):
            if not c.get(f"/sessions/{sid}").json()["has_video"]:
                break
            time.sleep(0.05)
        got = c.get(f"/sessions/{sid}").json()
        assert got["has_video"] is False and got["status"] == "done" and got["per_person"]   # numbers kept
        assert list((tmp_path / "up").iterdir()) == []


def test_t64_error_tracking_off_without_dsn_on_with_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_error_tracking() is False
    calls = {}
    fake = types.SimpleNamespace(init=lambda **kw: calls.update(kw))
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
    assert init_error_tracking() is True
    assert calls["dsn"].startswith("https://") and calls["send_default_pii"] is False


def test_t62_demo_sessions_load_and_have_results(tmp_path):
    store = Store(f"sqlite:///{(tmp_path / 'd.db').as_posix()}")
    ids = import_dir(store, ROOT / "demo" / "sessions")
    assert sorted(ids) == ["es2002a", "es2002b", "es2003a", "es2008a"]
    for sid in ids:
        s = store.get_session(sid)
        assert s["status"] == "done" and s["source"] == "ami" and len(s["per_person"]) == 4
        assert len(store.get_windows(sid)) > 50


def test_t62_static_demo_export(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_demo import export

    ids = export(ROOT / "demo" / "sessions", tmp_path)
    listing = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert {s["id"] for s in listing} == set(ids)
    for sid in ids:
        assert json.loads((tmp_path / "sessions" / f"{sid}.json").read_text(encoding="utf-8"))["status"] == "done"
        assert json.loads((tmp_path / "sessions" / sid / "windows.json").read_text(encoding="utf-8"))
        assert (tmp_path / "sessions" / sid / "people" / "P1.json").exists()


def test_deploy_config_is_consistent():
    docker = (ROOT / "deploy" / "huggingface" / "Dockerfile").read_text(encoding="utf-8")
    space = (ROOT / "deploy" / "huggingface" / "README.md").read_text(encoding="utf-8")
    assert "EXPOSE 7860" in docker and "--port ${PORT}" in docker and "PORT=7860" in docker
    assert "import_ami demo/sessions" in docker                          # demo data re-imported on boot
    assert "sdk: docker" in space and "app_port: 7860" in space
    for wf in ("ci.yml", "pages.yml", "deploy-hf.yml", "uptime.yml"):
        assert (ROOT / ".github" / "workflows" / wf).exists(), wf
    pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "VITE_DEMO" in pages and "export_demo.py" in pages
    uptime = (ROOT / ".github" / "workflows" / "uptime.yml").read_text(encoding="utf-8")
    assert '*/5 * * * *' in uptime and "/health" in uptime
