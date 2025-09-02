# Synthetic demo: FLI vs 3×2pt

This repository ships a fast, CPU-only synthetic baseline to compare
field-level inference (FLI) with a compressed 3×2pt-like analysis on the same
toy dataset. It runs in seconds on a laptop and serves as an executable
blueprint for the real analysis.

What it does
- Generates flat-sky 2D maps on an `nx × ny` grid:
  - Underlying Gaussian matter overdensity `δ` with a smooth toy power shape `P0(k)`.
  - Galaxy overdensity `g = b·δ + n_g` (white Gaussian pixel noise).
  - Shear E-mode `γ_E = 0.5·δ + n_γ` (white Gaussian pixel noise).
- Runs two inference paths for parameters `(A, b)`:
  - FLI: exact per-mode complex-Gaussian likelihood in Fourier space, marginalizing
    the latent field analytically (diagonal in k for a periodic box).
  - 3×2pt: binned FFT power spectra `P_gg`, `P_gE`, `P_EE` with a diagonal Gaussian
    likelihood using per-bin sample variances.
- Plots both posteriors together and marks the truth.

Quick start
- Install and test (see Getting Started page), then run:

```
python scripts/run_synthetic_comparison.py --config configs/demo.yaml
```

Outputs
- `figures/synthetic_constraints.png`: 2D contours (68/95%) and 1D marginals for A and b,
  overlaying FLI (C0) and 3×2pt (C1).
- Console logs with ML estimates for `(A, b)` for both methods.

Notes
- The model is intentionally simple, but the wiring mirrors the real pipeline:
  shared latent field, correlated observables, clear forward model, and apples-to-apples
  comparison between full-mode and compressed summaries.
- The amplitude `A` is an effective parameter (not calibrated to σ8). The comparison focuses
  on constraint sizes and relative performance, not cosmological realism.
