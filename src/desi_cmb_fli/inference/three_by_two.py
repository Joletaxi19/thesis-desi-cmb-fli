"""3×2pt-style binned power spectrum likelihood for flat-sky synthetic maps.

We estimate spectra P_gg, P_gE, P_EE in radial k-bins using FFTs and use an
approximate Gaussian likelihood on the binned estimates with variances
estimated from the scatter of per-mode values within each bin (divided by the
number of modes). This is deliberately simple for speed and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.power import bin_power, radial_binning, spectra_3x2pt
from ..sim.flat_sky import base_power_spectrum


@dataclass
class ThreeByTwoResult:
    A_grid: np.ndarray
    b_grid: np.ndarray
    nll: np.ndarray
    A_ML: float
    b_ML: float


def predict_spectra_bins(
    kk: np.ndarray,
    nbin: int,
    A: float,
    b: float,
    sigma_g: float,
    sigma_gamma: float,
    box_size: float,
    k0: float = 0.1,
    n: float = -2.0,
    kcut: float = 1.5,
):
    """Return predicted binned spectra under the toy model.

    P_gg = b^2 A P0 + N_g
    P_EE = 0.25 A P0 + N_gamma
    P_gE = 0.5 b A P0

    White noise adds a flat spectrum with power Npix σ^2 under numpy FFT.
    """
    nx = kk.shape[1]
    ny = kk.shape[0]
    P0 = base_power_spectrum(kk, k0=k0, n=n, kcut=kcut)
    P0[0, 0] = 0.0
    Npix = nx * ny
    Ng = Npix * (sigma_g**2)
    Ne = Npix * (sigma_gamma**2)
    # See notes in fli_flat: power carries an extra Npix factor under numpy FFT.
    P_field = A * P0 * Npix
    Pgg = (b**2) * P_field + Ng
    PEE = 0.25 * P_field + Ne
    PgE = 0.5 * b * P_field

    bins = radial_binning(kk, nbin=nbin)
    # Bin using the same mechanism as measurements for self-consistency.
    m_gg, _ = bin_power(Pgg, bins)
    m_gE, _ = bin_power(PgE, bins)
    m_EE, _ = bin_power(PEE, bins)
    return {"k": bins.centers, "Pgg": np.real(m_gg), "PgE": np.real(m_gE), "PEE": np.real(m_EE)}


def three_by_two_likelihood(
    g: np.ndarray,
    gammaE: np.ndarray,
    kk: np.ndarray,
    box_size: float,
    sigma_g: float,
    sigma_gamma: float,
    A_grid: np.ndarray,
    b_grid: np.ndarray,
    nbin: int = 15,
) -> ThreeByTwoResult:
    """Compute an approximate NLL over a grid for the binned 3×2pt spectra.

    The likelihood is a diagonal Gaussian over bins and spectra using variances
    estimated from the sample variance across modes in each bin.
    """
    spec_obs = spectra_3x2pt(g, gammaE, kk, nbin=nbin)
    # Stack observation vectors and variances
    spectra_names = ["Pgg", "PgE", "PEE"]
    y = np.concatenate([spec_obs[name]["mean"] for name in spectra_names])
    var = np.concatenate([spec_obs[name]["var"] for name in spectra_names])
    # Stabilize very small variances (empty bins already set to inf)
    var = np.where(var <= 0, np.inf, var)

    AA, BB = np.meshgrid(A_grid, b_grid, indexing="xy")
    nll = np.zeros_like(AA, dtype=np.float64)
    for ia in range(AA.shape[0]):
        for ib in range(AA.shape[1]):
            pred = predict_spectra_bins(
                kk,
                nbin=nbin,
                A=float(AA[ia, ib]),
                b=float(BB[ia, ib]),
                sigma_g=sigma_g,
                sigma_gamma=sigma_gamma,
                box_size=box_size,
            )
            mu = np.concatenate([pred[name] for name in spectra_names])
            r = y - mu
            nll[ia, ib] = 0.5 * np.sum((r**2) / var)

    min_idx = np.unravel_index(np.argmin(nll), nll.shape)
    A_ML = AA[min_idx]
    b_ML = BB[min_idx]
    return ThreeByTwoResult(
        A_grid=A_grid, b_grid=b_grid, nll=nll, A_ML=float(A_ML), b_ML=float(b_ML)
    )
