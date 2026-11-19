"""T0.3 (repo shape) and T0.5 (data licence credit)."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_licence_credit_in_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "AMI Meeting Corpus" in text
    assert "CC BY 4.0" in text


def test_repo_layout():
    for d in ["pipeline/grouppulse", "configs", "tests", "api", "web", "notebooks", "docker", "scripts"]:
        assert (ROOT / d).is_dir(), d
    for f in ["pyproject.toml", "Makefile", "scripts/setup.ps1", "docker/Dockerfile", ".github/workflows/ci.yml"]:
        assert (ROOT / f).is_file(), f


def test_pyproject_is_valid():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "grouppulse"
    assert "dev" in data["project"]["optional-dependencies"]
