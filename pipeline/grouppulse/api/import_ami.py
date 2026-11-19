"""Load pre-processed AMI session summaries into the database as demo sessions (test T6.2).

    python -m grouppulse.api.import_ami path/to/ami-data/sessions

Each <meeting>.json (from grouppulse.social.report --out) becomes a ready "AMI demo" session whose
id is the meeting id, so re-running replaces it instead of adding a duplicate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .settings import Settings
from .store import Store

TITLES = {"ES2002a": "Remote control design, kickoff", "ES2002b": "Remote control design, functional",
          "ES2003a": "Remote control design, kickoff (group 2)", "ES2008a": "Remote control design, kickoff (group 3)",
          "IS1009a": "Remote control design, kickoff (Idiap)", "IS1009b": "Remote control design, functional (Idiap)"}


def import_dir(store: Store, folder: Path) -> list[str]:
    done = []
    for p in sorted(Path(folder).glob("*.json")):
        summ = json.loads(p.read_text(encoding="utf-8"))
        mid = summ.get("session") or p.stem
        store.delete_session(mid.lower())
        sid = store.create_session(f"{mid} · {TITLES.get(mid, 'AMI meeting')}", "ami", session_id=mid.lower())
        store.save_summary(sid, summ)
        done.append(sid)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    ids = import_dir(Store(Settings().database_url), args.folder)
    print(f"imported {len(ids)} AMI demo sessions: {' '.join(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
