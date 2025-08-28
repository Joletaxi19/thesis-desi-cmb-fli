from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_toy_plot(value: float, out: str | Path) -> None:
    plt.figure()
    plt.title("Toy summary statistic")
    plt.plot([0, 1], [0, value])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
