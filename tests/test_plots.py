"""Tests for plotting helpers.

Uses a non-interactive backend and a temporary directory to verify that files
are created without depending on display availability.
"""

import matplotlib

# Set the backend before importing any module that pulls in pyplot.
matplotlib.use("Agg")


def test_save_toy_plot(tmp_path):
    """Saving a tiny plot produces a file on disk."""
    # Import after backend selection to avoid triggering an interactive backend.
    from desi_cmb_fli.analysis.plots import save_toy_plot

    out = tmp_path / "toy.png"
    save_toy_plot(0.3, out)
    assert out.exists()
