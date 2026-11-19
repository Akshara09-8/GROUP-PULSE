"""Splits are by meeting series and deterministic; config and download plan resolve correctly."""

from pathlib import Path

from grouppulse.data import load_config, series_of, split_meetings
from grouppulse.data.ami_download import plan


def test_series_of():
    assert series_of("ES2002a") == "ES2002"
    assert series_of("IS1009d") == "IS1009"
    assert series_of("custom") == "custom"


def test_split_keeps_series_together_and_is_stable():
    cfg = load_config()
    dev = cfg.meetings("dev")
    a = split_meetings(dev, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    b = split_meetings(dev, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    assert a == b
    assert sorted(sum(a.values(), [])) == sorted(dev)
    series_split = {}
    for split, ms in a.items():
        assert ms, f"{split} is empty"
        for m in ms:
            assert series_split.setdefault(series_of(m), split) == split


def test_config_sets_and_urls():
    cfg = load_config()
    assert len(cfg.meetings("demo")) == 5
    assert cfg.meetings(["demo", "ES2016a"])[-1] == "ES2016a"
    assert cfg.video_url_for("ES2002a", "Corner").endswith("/ES2002a/video/ES2002a.Corner.avi")


def test_room_view_depends_on_site():
    cfg = load_config()
    assert cfg.views_for("ES2002a")[-1] == "Corner"
    assert cfg.views_for("IS1009a")[-1] == "C"
    assert cfg.views_for("IS1009a", ["Closeup1", "room"]) == ["Closeup1", "C"]


def test_download_plan_paths():
    cfg = load_config()
    jobs = plan(["ES2002a"], ["Closeup1", "Corner"], True, Path("data/raw"), cfg)
    assert len(jobs) == 3
    assert jobs[0][1].name == "ami_public_manual_1.6.2.zip"
    assert jobs[2][1].as_posix() == "data/raw/amicorpus/ES2002a/video/ES2002a.Corner.avi"
