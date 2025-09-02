"""Smoke test for Matplotlib figure saving (non-interactive backend).

Ensures a figure can be created and saved to disk during tests/CI without
requiring a display.
"""

import matplotlib

# Select a non-interactive backend before importing pyplot.
matplotlib.use("Agg")


def test_can_save_figure(tmp_path):
    import matplotlib.pyplot as plt

    out = tmp_path / "toy.png"
    plt.figure()
    plt.plot([0, 1], [0, 0.3])
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

    assert out.exists()
    assert out.stat().st_size > 0
