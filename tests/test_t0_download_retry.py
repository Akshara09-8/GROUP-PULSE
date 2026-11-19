"""Download retries: back off on HTTP 403 (AMI rate limit), give up on 404."""

import urllib.error

import pytest

from grouppulse.data import ami_download


def _err(code):
    return urllib.error.HTTPError("http://x/y.avi", code, "msg", {}, None)


def test_retries_403_then_succeeds(monkeypatch, tmp_path):
    calls = []

    def fake(url, dest):
        calls.append(url)
        if len(calls) < 3:
            raise _err(403)
        return dest

    monkeypatch.setattr(ami_download, "download", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert ami_download.download_with_retry("http://x/y.avi", tmp_path / "y.avi", waits=(1, 1, 1)) == tmp_path / "y.avi"
    assert len(calls) == 3


def test_404_is_not_retried(monkeypatch, tmp_path):
    calls = []

    def fake(url, dest):
        calls.append(url)
        raise _err(404)

    monkeypatch.setattr(ami_download, "download", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        ami_download.download_with_retry("http://x/y.avi", tmp_path / "y.avi", waits=(1, 1))
    assert len(calls) == 1


def test_gives_up_after_last_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(ami_download, "download", lambda u, d: (_ for _ in ()).throw(_err(403)))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        ami_download.download_with_retry("http://x/y.avi", tmp_path / "y.avi", waits=(1, 1))
