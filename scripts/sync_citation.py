"""
Sync CITATION.cff fields from pyproject.toml.

Updates:
- version: set to project.version
- date-released: set to today if --update-date specified

Usage:
  python scripts/sync_citation.py --update-date
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

try:  # Python 3.11+
    # Built-in TOML parser in Python 3.11+.
    import tomllib as toml
except Exception:  # pragma: no cover - fallback for older Pythons
    # Fallback to external library on older interpreters.
    import tomli as toml  # type: ignore

import yaml

# Repository root (assumes script lives in ./scripts).
ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict:
    data = toml.loads((ROOT / "pyproject.toml").read_text())
    return data.get("project", {})


def sync_citation(update_date: bool = False) -> bool:
    project = load_pyproject()
    version = project.get("version")
    if not version:
        raise RuntimeError("project.version not found in pyproject.toml")

    cff_path = ROOT / "CITATION.cff"
    cff = yaml.safe_load(cff_path.read_text())

    changed = False
    if cff.get("version") != version:
        cff["version"] = version
        changed = True

    if update_date:
        today = _dt.date.today().isoformat()
        if cff.get("date-released") != today:
            cff["date-released"] = today
            changed = True

    if changed:
        # Keep original key order for readability in reviews/diffs.
        cff_path.write_text(yaml.safe_dump(cff, sort_keys=False))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-date", action="store_true", help="Also update date-released to today")
    # Pre-commit may pass filenames; accept and ignore them gracefully
    ap.add_argument("paths", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()
    changed = sync_citation(update_date=args.update_date)
    if changed:
        print("CITATION.cff updated")
    else:
        print("CITATION.cff already in sync")


if __name__ == "__main__":
    main()
