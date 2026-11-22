"""Phase 3 tests T3.1-T3.7 (+ retention T6.3 logic, AMI import T6.2) against a temp SQLite store."""

import io
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from grouppulse.api import Settings, create_app  # noqa: E402
from grouppulse.api.import_ami import import_dir  # noqa: E402
from grouppulse.api.store import Store  # noqa: E402
from grouppulse.api.worker import ProcessingUnavailable, fail_orphans  # noqa: E402
from grouppulse.perception.io import TrackRow  # noqa: E402
from grouppulse.social import analyze  # noqa: E402

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096


def summary(people: int, seconds: int = 30) -> dict:
    labels = [f"P{i + 1}" for i in range(people)]
    rows = [TrackRow("UP", k * 5, k / 5, p, True, 0.0, 0.0, None, labels[0] if p != labels[0] else labels[1],
                     None, None, 0.0, False) for k in range(seconds * 5) for p in labels]
    return analyze(rows, None)


def make(tmp_path, processor=None, **kw):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'gp.db').as_posix()}", files_dir=tmp_path / "up", **kw)
    return create_app(s, processor=processor or (lambda video, people: summary(people)))


def post(c, data=MP4, name="m.mp4", people=4, consent=True, title="Kickoff"):
    return c.post("/sessions", files={"file": (name, io.BytesIO(data), "video/mp4")},
                  data={"people": str(people), "consent": str(consent).lower(), "title": title})


def wait_status(c, sid, want, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/sessions/{sid}").json()
        if s["status"] == want:
            return s
        time.sleep(0.02)
    raise AssertionError(f"status stayed {s['status']}, wanted {want}")


def test_t31_upload_is_queued_fast(tmp_path):
    with TestClient(make(tmp_path)) as c:
        t0 = time.time()
        r = post(c)
        assert r.status_code == 202 and time.time() - t0 < 3
        assert r.json()["status"] == "queued" and len(r.json()["id"]) == 12


def test_t32_status_moves_through_states(tmp_path):
    gate = threading.Event()

    def slow(video, people):
        gate.wait(5)
        return summary(people)

    with TestClient(make(tmp_path, processor=slow)) as c:
        sid = post(c).json()["id"]
        wait_status(c, sid, "processing")
        gate.set()
        done = wait_status(c, sid, "done")
        assert done["group"]["dominant"] == "P1"


def test_t32_failures_are_reported_not_stuck(tmp_path):
    def broken(video, people):
        raise ValueError("boom")

    with TestClient(make(tmp_path, processor=broken)) as c:
        s = wait_status(c, post(c).json()["id"], "failed")
        assert s["error"] == "Processing failed: ValueError"

    def unsupported(video, people):
        raise ProcessingUnavailable("Uploads from a single camera are not supported yet.")

    with TestClient(make(tmp_path / "b", processor=unsupported)) as c:
        s = wait_status(c, post(c).json()["id"], "failed")
        assert "not supported yet" in s["error"]


def test_t33_results_read_back_exactly(tmp_path):
    with TestClient(make(tmp_path)) as c:
        sid = post(c, people=3).json()["id"]
        s = wait_status(c, sid, "done")
        want = summary(3)
        assert s["per_person"] == want["per_person"]
        stored = {k: v for k, v in want["group"].items() if k not in ("attention_method", "attention_raw")}
        assert s["group"] == stored                      # diagnostics are not persisted
        assert c.get(f"/sessions/{sid}/windows").json() == want["windows"]
        p = c.get(f"/sessions/{sid}/people/P2").json()
        assert p["dominance"] == want["per_person"]["P2"]["dominance"]
        assert len(p["engagement_over_time"]) == len(want["windows"])


def test_t34_invalid_uploads_rejected_and_nothing_stored(tmp_path):
    with TestClient(make(tmp_path, max_upload_mb=1)) as c:
        assert post(c, data=b"hello, this is text", name="notes.txt").status_code == 415
        assert post(c, data=b"hello, this is text", name="fake.mp4").status_code == 415   # checked by content
        assert post(c, data=MP4 + b"\x00" * (2 << 20)).status_code == 413
        assert post(c, consent=False).status_code == 400
        assert post(c, people=9).status_code == 422
        assert c.get("/sessions").json() == []
        assert list((tmp_path / "up").iterdir()) == []


def test_t35_two_uploads_at_once_keep_their_own_results(tmp_path):
    with TestClient(make(tmp_path)) as c:
        a, b = post(c, people=3).json()["id"], post(c, people=5).json()["id"]
        sa, sb = wait_status(c, a, "done"), wait_status(c, b, "done")
        assert sorted(sa["per_person"]) == ["P1", "P2", "P3"]
        assert sorted(sb["per_person"]) == ["P1", "P2", "P3", "P4", "P5"]


def test_t36_demo_full_then_retention_frees_space(tmp_path):
    with TestClient(make(tmp_path, max_active_uploads=1, retention_hours=1e-6)) as c:
        first = post(c).json()["id"]
        wait_status(c, first, "done")
        time.sleep(0.01)
        # the first video has already expired: upload purges it, so this one fits
        r = post(c)
        assert r.status_code == 202
        s = c.get(f"/sessions/{first}").json()
        assert s["has_video"] is False and s["status"] == "done" and s["per_person"]   # numbers kept (T6.3)
    with TestClient(make(tmp_path / "full", max_active_uploads=1)) as c:
        assert post(c).status_code == 202
        r = post(c)
        assert r.status_code == 507 and "try again later" in r.json()["detail"]


def test_t37_docs_list_all_endpoints(tmp_path):
    with TestClient(make(tmp_path)) as c:
        assert c.get("/docs").status_code == 200
        paths = c.get("/openapi.json").json()["paths"]
        for p in ("/health", "/sessions", "/sessions/{sid}", "/sessions/{sid}/windows", "/sessions/{sid}/people/{label}"):
            assert p in paths


def test_delete_removes_results_and_video(tmp_path):
    with TestClient(make(tmp_path)) as c:
        sid = post(c).json()["id"]
        wait_status(c, sid, "done")
        assert c.delete(f"/sessions/{sid}").status_code == 204
        assert c.get(f"/sessions/{sid}").status_code == 404
        assert list((tmp_path / "up").iterdir()) == []
        assert c.delete(f"/sessions/{sid}").status_code == 404


def test_orphaned_jobs_fail_on_restart(tmp_path):
    store = Store(f"sqlite:///{(tmp_path / 'o.db').as_posix()}")
    sid = store.create_session("x", "upload", 4, "v.mp4", 10, 24)
    store.set_status(sid, "processing")
    assert fail_orphans(store) == 1
    assert store.get_session(sid)["status"] == "failed"


def test_t62_ami_import(tmp_path):
    folder = tmp_path / "sessions"
    folder.mkdir()
    (folder / "ES2002a.json").write_text(json.dumps({"session": "ES2002a", **summary(4)}), encoding="utf-8")
    store = Store(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
    assert import_dir(store, folder) == ["es2002a"]
    assert import_dir(store, folder) == ["es2002a"]                    # re-import replaces, no duplicate
    s = store.get_session("es2002a")
    assert s["source"] == "ami" and s["status"] == "done" and s["title"].startswith("ES2002a")
    assert len(store.list_sessions()) == 1
