"""Fast flat-sky synthetic data generator.

This module produces small 2D maps (CPU-friendly) for a toy joint dataset
consisting of a biased galaxy overdensity map and a weak-lensing E-mode map.
Both derive from an underlying Gaussian density field so they are correlated,
enabling a simple 3×2pt-like analysis and a field-level (Fourier-mode) likelihood.

Design goals:
- Minimal dependencies (NumPy FFTs) and no cosmology libraries.
- Deterministic PRNG via ``numpy.random.Generator``.
- Defaults tuned to run in seconds on a laptop CPU (e.g. 128×128).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthConfig:
    """Configuration for synthetic map generation.

    Attributes
    ----------
    nx, ny : int
        Number of pixels along x and y (square pixels assumed).
    box_size : float
        Physical size of the box (arbitrary units). Used only to set the
        fundamental Fourier frequencies; absolute units cancel in the demo.
    A : float
        Amplitude of the underlying density power spectrum (acts like sigma8^2
        in spirit but not calibrated to a real cosmology).
    b : float
        Linear galaxy bias relating galaxy overdensity to matter overdensity.
    sigma_g : float
        Per-pixel Gaussian noise std. dev. for the galaxy map.
    sigma_gamma : float
        Per-pixel Gaussian noise std. dev. for the shear E-mode map.
    seed : int
        Random seed for reproducibility.
    """

    nx: int = 128
    ny: int = 128
    box_size: float = 400.0
    A: float = 1.0
    b: float = 1.2
    sigma_g: float = 0.5
    sigma_gamma: float = 0.3
    seed: int = 0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"Grid dimensions must be positive: nx={self.nx}, ny={self.ny}")
        if self.box_size <= 0:
            raise ValueError(f"Box size must be positive: {self.box_size}")
        if self.A < 0:
            raise ValueError(f"Amplitude A must be non-negative: {self.A}")
        if self.sigma_g < 0 or self.sigma_gamma < 0:
            raise ValueError(
                f"Noise levels must be non-negative: sigma_g={self.sigma_g}, sigma_gamma={self.sigma_gamma}"
            )
        if self.seed < 0:
            raise ValueError(f"Random seed must be non-negative: {self.seed}")


def _kgrid(nx: int, ny: int, box_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2D Fourier wavevectors and their magnitude on a regular grid.

    Uses numpy.fft frequency conventions. The returned arrays have shape (ny, nx).
    """
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=box_size / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=box_size / ny)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(kx2d**2 + ky2d**2)
    return kx2d, ky2d, kk


def base_power_spectrum(
    k: np.ndarray, k0: float = 0.1, n: float = -2.0, kcut: float = 1.5
) -> np.ndarray:
    """Toy isotropic power spectrum shape P0(k).

    The functional form is chosen for smoothness and finiteness at k→0 and k→∞.
    P0(k) ∝ (k + k0)^n × exp(-(k/kcut)^2)
    """
    return (k + k0) ** n * np.exp(-((k / kcut) ** 2))


def generate_maps(cfg: SynthConfig) -> dict:
    """Generate correlated galaxy overdensity and shear E-mode maps.

    Returns a dictionary with keys:
    - "delta": underlying matter overdensity map
    - "g": galaxy overdensity map (b·delta + noise)
    - "gammaE": shear E-mode map (0.5·delta + noise)
    - "kk": 2D array of |k| for the grid (useful for later binning)
    - "cfg": the config used (echoed back for convenience)
    """
    rng = np.random.default_rng(cfg.seed)
    nx, ny = cfg.nx, cfg.ny
    kx, ky, kk = _kgrid(nx, ny, cfg.box_size)

    # Draw real-space white noise and filter it in Fourier space to imprint P(k).
    w = rng.normal(size=(ny, nx))
    W = np.fft.fft2(w)
    P0 = base_power_spectrum(kk)
    # Enforce zero power at the DC mode to avoid degeneracies.
    P0[0, 0] = 0.0
    # Scale white-noise modes by sqrt(A · P0(k)) to obtain the target spectrum.
    Dk = np.sqrt(np.maximum(cfg.A * P0, 0.0)) * W
    delta = np.fft.ifft2(Dk).real

    # Observables: galaxy overdensity and shear E-mode.
    g = cfg.b * delta + rng.normal(scale=cfg.sigma_g, size=delta.shape)
    gammaE = 0.5 * delta + rng.normal(scale=cfg.sigma_gamma, size=delta.shape)

    return {"delta": delta, "g": g, "gammaE": gammaE, "kk": kk, "cfg": cfg}
