"""Read configs/ami.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ami.toml"


@dataclass(frozen=True)
class AmiConfig:
    annotations_url: str
    video_url: str
    fps: float
    views: tuple[str, ...]
    room_views: dict[str, tuple[str, ...]]
    meeting_sets: dict[str, tuple[str, ...]]
    seed: int
    val_fraction: float
    test_fraction: float

    def meetings(self, name_or_ids: str | list[str]) -> list[str]:
        """Resolve a set name ("demo", "dev") or an explicit list of meeting ids."""
        if isinstance(name_or_ids, str):
            if name_or_ids in self.meeting_sets:
                return list(self.meeting_sets[name_or_ids])
            return [name_or_ids]
        out: list[str] = []
        for item in name_or_ids:
            out.extend(self.meetings(item))
        return out

    def views_for(self, meeting: str, views: list[str] | tuple[str, ...] | None = None) -> list[str]:
        """Camera views for a meeting, with "room" expanded to that site's room cameras (ES -> Corner, IS -> C)."""
        out: list[str] = []
        for v in (views if views is not None else self.views):
            if v == "room":
                out.extend(self.room_views.get(meeting[:2], ("Corner",)))
            elif v not in out:
                out.append(v)
        return out

    def video_url_for(self, meeting: str, view: str) -> str:
        return self.video_url.format(meeting=meeting, view=view)


def load_config(path: str | Path | None = None) -> AmiConfig:
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open("rb") as f:
        raw = tomllib.load(f)
    return AmiConfig(
        annotations_url=raw["source"]["annotations_url"],
        video_url=raw["source"]["video_url"],
        fps=float(raw["video"]["fps"]),
        views=tuple(raw["video"]["views"]),
        room_views={k: tuple(v) for k, v in raw["video"].get("room_views", {}).items()},
        meeting_sets={k: tuple(v) for k, v in raw["meetings"].items()},
        seed=int(raw["splits"]["seed"]),
        val_fraction=float(raw["splits"]["val_fraction"]),
        test_fraction=float(raw["splits"]["test_fraction"]),
    )
