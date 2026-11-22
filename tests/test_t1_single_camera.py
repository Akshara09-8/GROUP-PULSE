"""Single-camera upload path (perception/single.py) and its overlay in the API."""

import io
import json
import time

import pytest

from grouppulse.perception.single import FaceObs, assemble, assign_people, image_gaze_target, name_tracks

# Four people around a table seen from one camera: two on the left, two on the right.
HEADS = {"P1": (100, 200), "P2": (220, 150), "P3": (420, 150), "P4": (540, 200)}


def test_gaze_target_geometry():
    # Smallest angle to the gaze line wins; with no depth, a head further along that line can beat a nearer one.
    assert image_gaze_target("P1", HEADS, 60, 0) == "P4"        # level gaze to the right: P4 is on the line
    assert image_gaze_target("P4", HEADS, -55, 0) == "P1"
    assert image_gaze_target("P2", HEADS, 45, 0) == "P3"        # P3 sits at P2's height, straight across
    assert image_gaze_target("P2", HEADS, 0, 0) == "away"       # facing the camera: nobody in that direction
    assert image_gaze_target("P3", HEADS, 30, -30) == "table"   # looking down
    assert image_gaze_target("P1", HEADS, None, 0) == "unknown"
    assert image_gaze_target("P1", HEADS, -60, 0) == "away"     # nobody to P1's left


def obs_for(track, x, frames, yaw=0.0):
    return [FaceObs(f, f / 25, track, (x - 20, 180, x + 20, 220), yaw, 0.0) for f in frames]


def test_tracks_named_left_to_right_and_fragments_reattached():
    frames = range(0, 100, 10)
    obs = obs_for(7, 500, frames) + obs_for(3, 100, frames) + obs_for(9, 300, frames[:5])
    obs += obs_for(42, 305, frames[5:])                          # the tracker lost P2 and gave a new id
    obs += obs_for(99, 900, [50])                               # a passer-by far from every seat
    assert set(name_tracks(obs, 3).values()) == {"P1", "P2", "P3"}
    seats = assign_people(obs, 3)
    assert all(set(seats[f]) == {"P1", "P2", "P3"} for f in frames)
    assert seats[60]["P2"].track == 42 and seats[60]["P1"].track == 3


def test_assemble_rows_and_overlay():
    frames = range(0, 50, 10)
    obs = obs_for(1, 100, frames, yaw=60) + obs_for(2, 400, frames, yaw=-60)
    rows, ov = assemble(obs, 2, (640, 480), 25.0, 10, session="up1")
    assert {r.person for r in rows} == {"P1", "P2"} and len(rows) == 10
    assert {r.gaze_target for r in rows} == {"P1", "P2"}         # they look at each other
    assert ov["people"] == ["P1", "P2"] and ov["fps"] == 2.5
    first = ov["frames"][0]["people"]["P1"]
    assert first["target"] == "P2" and first["box"] == [0.125, 0.375, 0.1875, 0.4583]


def test_api_serves_video_and_overlay_then_deletes_both(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from grouppulse.api import Settings, create_app
    from grouppulse.perception.io import TrackRow
    from grouppulse.social import analyze

    def proc(video, people):
        rows = [TrackRow("x", k, k / 5, p, True, 0.0, 0.0, None, "P2" if p == "P1" else "P1", None, None, 0.0, False)
                for k in range(50) for p in ("P1", "P2", "P3")]
        s = analyze(rows, None)
        s["overlay"] = {"fps": 5, "people": ["P1", "P2", "P3"], "frames": [{"t": 0, "people": {}}]}
        return s

    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'g.db').as_posix()}", files_dir=tmp_path / "up")
    with TestClient(create_app(s, processor=proc)) as c:
        sid = c.post("/sessions", files={"file": ("m.mp4", io.BytesIO(mp4), "video/mp4")},
                     data={"people": "3", "consent": "true"}).json()["id"]
        for _ in range(200):
            if c.get(f"/sessions/{sid}").json()["status"] == "done":
                break
            time.sleep(0.02)
        assert c.get(f"/sessions/{sid}/overlay").json()["people"] == ["P1", "P2", "P3"]
        v = c.get(f"/sessions/{sid}/video")
        assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and v.content == mp4
        assert c.delete(f"/sessions/{sid}").status_code == 204
        assert list((tmp_path / "up").iterdir()) == []                 # video and overlay both gone
        assert c.get("/sessions/zzz/video").status_code == 404
