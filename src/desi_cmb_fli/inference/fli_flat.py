"""Field-level (Fourier-mode) likelihood for flat-sky synthetic maps.

We model the observed complex Fourier coefficients of two fields y_k = [g_k, gammaE_k]
as zero-mean proper complex Gaussians with a 2×2 covariance that depends on
parameters theta = (A, b):

    C_delta(k) = A · P0(k)
    S_11 = b^2 · C_delta(k) + N_g
    S_22 = 0.25 · C_delta(k) + N_gamma
    S_12 = 0.5 b · C_delta(k)

Noise levels N_g, N_gamma are constant in Fourier space for white pixel noise
and equal to Npix · sigma^2 under NumPy FFT conventions.

We restrict the sum to a set of unique modes to avoid double-counting conjugate
pairs, and exclude the DC mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.flat_sky import base_power_spectrum  # reuse same P0(k) shape


def unique_halfplane_indices(nx: int, ny: int) -> np.ndarray:
    """Return flat indices of unique Fourier modes (exclude DC and conjugates).

    We include modes with ky > 0 and the ky = 0 line with kx > 0.
    """
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    mask = (KY > 0) | ((KY == 0) & (KX > 0))
    mask_flat = mask.ravel()
    idx = np.where(mask_flat)[0]
    return idx


@dataclass
class FLIFourierResult:
    A_grid: np.ndarray
    b_grid: np.ndarray
    nll: np.ndarray  # negative log-likelihood grid
    A_ML: float
    b_ML: float


def fli_fourier_likelihood(
    g: np.ndarray,
    gammaE: np.ndarray,
    box_size: float,
    sigma_g: float,
    sigma_gamma: float,
    A_grid: np.ndarray,
    b_grid: np.ndarray,
    k0: float = 0.1,
    n: float = -2.0,
    kcut: float = 1.5,
) -> FLIFourierResult:
    """Compute NLL over a parameter grid for the field-level (mode) likelihood.

    Parameters
    ----------
    g, gammaE : ndarray
        Real-space maps with identical shape (ny, nx).
    box_size : float
        Physical size used for k-grid construction (same as generator).
    sigma_g, sigma_gamma : float
        Pixel noise standard deviations used to set constant Fourier noise.
    A_grid, b_grid : ndarray
        1D arrays of parameter values to scan.
    k0, n, kcut : float
        Parameters of the base power shape P0(k). Must match generator.
    """
    ny, nx = g.shape
    # Fourier transforms of the observed maps.
    Gk = np.fft.fft2(g)
    Ek = np.fft.fft2(gammaE)

    # Build k-magnitude array and power-shape P0(k)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=box_size / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=box_size / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KK = np.sqrt(KX**2 + KY**2)
    P0 = base_power_spectrum(KK, k0=k0, n=n, kcut=kcut)
    P0[0, 0] = 0.0

    # Unique mode selection to avoid conjugate double counting.
    idx = unique_halfplane_indices(nx, ny)
    Gk_u = Gk.ravel()[idx]
    Ek_u = Ek.ravel()[idx]
    P0_u = P0.ravel()[idx]

    # Constant white-noise power per mode under numpy FFT convention.
    Npix = nx * ny
    Ng = Npix * (sigma_g**2)
    Ne = Npix * (sigma_gamma**2)

    AA, BB = np.meshgrid(A_grid, b_grid, indexing="xy")
    nll = np.zeros_like(AA, dtype=np.float64)

    # For each grid point, accumulate the 2×2 complex Gaussian NLL across modes.
    for ia in range(AA.shape[0]):
        for ib in range(AA.shape[1]):
            A = AA[ia, ib]
            b = BB[ia, ib]
            # Under numpy FFT conventions, if the real-space white noise has unit
            # variance, the Fourier-space power picks up a factor Npix. Our synthetic
            # generator uses delta_k = sqrt(A P0) * W_k with W_k = FFT(white), hence
            # E[|delta_k|^2] = A P0 · E[|W_k|^2] = A P0 · Npix.
            Cdelta = A * P0_u * Npix  # per-mode underlying power
            s11 = (b**2) * Cdelta + Ng
            s22 = 0.25 * Cdelta + Ne
            s12 = 0.5 * b * Cdelta
            # Determinant and inverse for 2×2 Hermitian covariance
            det = s11 * s22 - np.abs(s12) ** 2
            inv11 = s22 / det
            inv22 = s11 / det
            inv12 = -s12 / det
            # Quadratic form y^H S^{-1} y for complex 2-vector [Gk, Ek]
            qf = (
                inv11 * (np.abs(Gk_u) ** 2)
                + inv22 * (np.abs(Ek_u) ** 2)
                + 2 * np.real(inv12 * (Gk_u * np.conj(Ek_u)))
            )
            # Proper complex Gaussian: log det factor per mode for 2-dim vector
            # Ignoring constant pi terms since we compare within the grid.
            ll = 0.5 * (np.log(det) + qf)
            nll[ia, ib] = np.sum(ll)

    # Locate ML point
    min_idx = np.unravel_index(np.argmin(nll), nll.shape)
    A_ML = AA[min_idx]
    b_ML = BB[min_idx]
    return FLIFourierResult(
        A_grid=A_grid, b_grid=b_grid, nll=nll, A_ML=float(A_ML), b_ML=float(b_ML)
    )
