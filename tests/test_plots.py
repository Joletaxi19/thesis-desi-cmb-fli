import matplotlib
matplotlib.use("Agg")

from desi_cmb_fli.analysis.plots import save_toy_plot


def test_save_toy_plot(tmp_path):
    out = tmp_path / "toy.png"
    save_toy_plot(0.3, out)
    assert out.exists()
