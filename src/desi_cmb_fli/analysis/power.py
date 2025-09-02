"""FFT-based power spectrum estimators for flat-sky maps.

This provides minimal, dependency-free (NumPy-only) estimators for auto- and
cross-power spectra along with simple radial binning utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Binning:
    edges: np.ndarray  # shape (nbin + 1,)
    centers: np.ndarray  # shape (nbin,)
    indices: list[np.ndarray]  # list of arrays of flat indices per bin


def radial_binning(
    kk: np.ndarray, nbin: int = 15, kmin: float | None = None, kmax: float | None = None
) -> Binning:
    """Build radial bins in |k| and index modes per bin.

    Parameters
    ----------
    kk : ndarray
        2D array of |k| magnitudes.
    nbin : int
        Number of radial bins.
    kmin, kmax : float | None
        Optional limits; defaults use min/max from kk excluding the DC mode.
    """
    km = kk.copy().ravel()
    # Exclude the DC mode at index 0.
    if kmin is None:
        kmin = float(km[km > 0].min())
    if kmax is None:
        kmax = float(km.max())
    edges = np.linspace(kmin, kmax, nbin + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices: list[np.ndarray] = []
    km2d = kk
    flat = km2d.ravel()
    for i in range(nbin):
        m = (flat >= edges[i]) & (flat < edges[i + 1])
        # remove the DC mode if included due to numerical boundaries
        m[np.where(flat == 0.0)[0]] = False
        idx = np.where(m)[0]
        indices.append(idx)
    return Binning(edges=edges, centers=centers, indices=indices)


def power_2d(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Return 2D cross-power P_ab(kx,ky) from two real maps.

    Uses NumPy FFT conventions. If b is None, computes auto-power P_aa.
    """
    Fa = np.fft.fft2(a)
    Fb = Fa if b is None else np.fft.fft2(b)
    P = Fa * np.conj(Fb)
    return P


def bin_power(Pk: np.ndarray, binning: Binning) -> tuple[np.ndarray, np.ndarray]:
    """Bin a 2D power array over radial shells and return mean and variance.

    Returns
    -------
    mean : ndarray
        Mean of power in each bin (complex for cross-power; real for auto).
    var : ndarray
        Variance of the per-mode power in each bin divided by N_modes, i.e.
        an estimate of Var(mean). For complex cross-power, the variance is
        computed on the real part which is the usual estimator for isotropic
        fields.
    """
    flat = Pk.ravel()
    nbin = len(binning.indices)
    mean = np.zeros(nbin, dtype=np.complex128)
    var = np.zeros(nbin, dtype=np.float64)
    for i, idx in enumerate(binning.indices):
        if idx.size == 0:
            mean[i] = 0.0
            var[i] = np.inf
            continue
        vals = flat[idx]
        m = vals.mean()
        mean[i] = m
        # Use real part for variance estimate (unbiased for isotropic fields)
        if idx.size == 1:
            # For single-element bins, set variance to zero (no variance estimate possible)
            v = 0.0
        else:
            v = np.var(np.real(vals), ddof=1) / idx.size
        var[i] = v
    return mean, var


def spectra_3x2pt(
    g: np.ndarray, gammaE: np.ndarray, kk: np.ndarray, nbin: int = 15
) -> dict[str, dict]:
    """Compute binned 3×2pt-like spectra: Pgg, PgE, PE E.

    Returns a dict mapping spectrum name to a dict with keys: centers, mean, var.
    """
    bins = radial_binning(kk, nbin=nbin)
    Pgg = power_2d(g)
    PgE = power_2d(g, gammaE)
    PEE = power_2d(gammaE)
    m_gg, v_gg = bin_power(Pgg, bins)
    m_gE, v_gE = bin_power(PgE, bins)
    m_EE, v_EE = bin_power(PEE, bins)
    return {
        "Pgg": {"k": bins.centers, "mean": np.real(m_gg), "var": v_gg},
        "PgE": {"k": bins.centers, "mean": np.real(m_gE), "var": v_gE},
        "PEE": {"k": bins.centers, "mean": np.real(m_EE), "var": v_EE},
    }
