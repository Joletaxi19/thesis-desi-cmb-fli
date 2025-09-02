"""Smoke tests for the synthetic FLI vs 3×2pt pipeline.

Kept tiny to ensure fast CI runtime on CPU.
"""

import numpy as np

from desi_cmb_fli.inference.fli_flat import fli_fourier_likelihood
from desi_cmb_fli.inference.three_by_two import three_by_two_likelihood
from desi_cmb_fli.sim.flat_sky import SynthConfig, generate_maps


def test_generate_maps_shapes():
    cfg = SynthConfig(nx=32, ny=32, seed=42)
    data = generate_maps(cfg)
    assert data["g"].shape == (32, 32)
    assert data["gammaE"].shape == (32, 32)
    assert data["kk"].shape == (32, 32)


def test_inference_runs_and_is_reasonable():
    cfg = SynthConfig(nx=32, ny=32, A=1.0, b=1.3, seed=0)
    d = generate_maps(cfg)
    A_grid = np.linspace(0.5, 1.5, 10)
    b_grid = np.linspace(0.8, 1.8, 10)

    # Field-level likelihood
    res_fli = fli_fourier_likelihood(
        d["g"],
        d["gammaE"],
        box_size=cfg.box_size,
        sigma_g=cfg.sigma_g,
        sigma_gamma=cfg.sigma_gamma,
        A_grid=A_grid,
        b_grid=b_grid,
    )
    # 3x2pt likelihood
    res_3x2 = three_by_two_likelihood(
        d["g"],
        d["gammaE"],
        d["kk"],
        box_size=cfg.box_size,
        sigma_g=cfg.sigma_g,
        sigma_gamma=cfg.sigma_gamma,
        A_grid=A_grid,
        b_grid=b_grid,
        nbin=8,
    )
    # ML points should be finite and near the truth (very weak check)
    assert np.isfinite(res_fli.A_ML)
    assert np.isfinite(res_fli.b_ML)
    assert np.isfinite(res_3x2.A_ML)
    assert np.isfinite(res_3x2.b_ML)
    assert abs(res_fli.A_ML - cfg.A) < 0.5
    assert abs(res_fli.b_ML - cfg.b) < 0.5
