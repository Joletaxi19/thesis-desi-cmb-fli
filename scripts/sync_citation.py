"""
Sync CITATION.cff fields from pyproject.toml.

Updates:
- version: set to project.version
- date-released: automatically updated to today when version changes

Usage:
  python scripts/sync_citation.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

try:
    # Built-in TOML parser in Python 3.11+
    import tomllib as toml
except ImportError:  # pragma: no cover - fallback for older Pythons
    # Fallback to external library on older interpreters
    try:
        import tomli as toml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Neither tomllib (Python 3.11+) nor tomli is available. "
            "Install tomli with 'pip install tomli'."
        ) from exc

import yaml

# Repository root (assumes script lives in ./scripts).
ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict:
    data = toml.loads((ROOT / "pyproject.toml").read_text())
    return data.get("project", {})


def sync_citation() -> bool:
    project = load_pyproject()
    version = project.get("version")
    if not version:
        raise RuntimeError("project.version not found in pyproject.toml")

    cff_path = ROOT / "CITATION.cff"
    cff = yaml.safe_load(cff_path.read_text())

    changed = False
    version_changed = False

    # Check if version needs updating
    if cff.get("version") != version:
        cff["version"] = version
        version_changed = True
        changed = True

    # If version changed, also update the release date
    if version_changed:
        today = _dt.date.today().isoformat()
        cff["date-released"] = today
        changed = True

    if changed:
        # Keep original key order for readability in reviews/diffs.
        cff_path.write_text(yaml.safe_dump(cff, sort_keys=False))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    # Pre-commit may pass filenames; accept and ignore them gracefully
    ap.add_argument("paths", nargs="*", help=argparse.SUPPRESS)
    ap.parse_args()

    changed = sync_citation()
    if changed:
        print("✓ CITATION.cff updated (version and date)")
    else:
        print("✓ CITATION.cff already in sync")


if __name__ == "__main__":
    main()
