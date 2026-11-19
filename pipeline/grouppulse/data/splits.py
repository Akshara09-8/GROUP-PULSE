"""Train / validation / test splits by meeting series.

AMI scenario meetings come in series of four (ES2002a-d) recorded by the same four people.
Putting ES2002a in train and ES2002b in test would let a model memorise those people, so a
whole series always goes to one split.
"""

from __future__ import annotations

import random
import re

_SERIES = re.compile(r"^([A-Z]{2}\d{4})[a-z]?$")


def series_of(meeting: str) -> str:
    m = _SERIES.match(meeting)
    return m.group(1) if m else meeting


def split_meetings(meetings: list[str], val_fraction: float = 0.2, test_fraction: float = 0.2,
                   seed: int = 13) -> dict[str, list[str]]:
    """Deterministic split. Every split gets at least one series when there are 3 or more."""
    series = sorted({series_of(m) for m in meetings})
    rng = random.Random(seed)
    rng.shuffle(series)
    n = len(series)
    n_test = max(1, round(n * test_fraction)) if n >= 3 else 0
    n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
    test, val = set(series[:n_test]), set(series[n_test:n_test + n_val])
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for m in sorted(meetings):
        s = series_of(m)
        out["test" if s in test else "val" if s in val else "train"].append(m)
    return out
