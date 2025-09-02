"""Lightweight plotting helpers for analysis outputs.

The goal of this module is to keep plotting utilities minimal and explicit so
they are easy to read and adapt. Each function should do exactly one thing and
save figures deterministically (including creating parent directories).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_toy_plot(value: float, out: str | Path) -> None:
    """Save a tiny line plot visualizing a single scalar statistic.

    Parameters
    ----------
    value
        The scalar value to illustrate.
    out
        Output path for the saved figure. Parent directories are created if
        they do not exist.
    """
    plt.figure()
    plt.title("Scalar summary statistic")
    # Simple 2-point line from (0, 0) to (1, value) for a quick visual cue.
    plt.plot([0, 1], [0, value])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Use tight bounding box and a moderate DPI for crisp but small files.
    plt.savefig(out, bbox_inches="tight", dpi=150)
    # Explicitly close to avoid leaking figure handles in batch runs/tests.
    plt.close()
