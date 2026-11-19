"""T0.1 (video part): frame count matches duration. Skipped unless OpenCV is installed."""

import pytest

from grouppulse.data.video import VideoInfo, frames_match_duration, probe


def test_frames_match_duration_logic(tmp_path):
    info = VideoInfo(tmp_path / "x.avi", frames=25 * 60, fps=25.0)
    assert info.duration == 60.0
    assert frames_match_duration(info, 60.0)
    assert frames_match_duration(info, 60.06)       # 1.5 frames off: fine
    assert not frames_match_duration(info, 61.0)    # 25 frames off: wrong file or truncated


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.avi")


def test_probe_real_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for i in range(50):
        w.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    w.release()
    info = probe(path)
    assert info.fps == pytest.approx(25.0)
    assert frames_match_duration(info, 2.0)
