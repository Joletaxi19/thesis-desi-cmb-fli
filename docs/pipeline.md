# Pipeline

High-level stages (real analysis):
1. **Data ingestion & map-making**
2. **Masks, beams, noise**
3. **Differentiable simulation (JAX)**
4. **Likelihood / posterior & FLI**
5. **Validation & null tests**
6. **Paper-ready plots**

Synthetic baseline (this repo):
- Flat-sky generator for correlated galaxy overdensity and shear E-mode maps.
- Two inference branches on identical data:
  - Field-level (Fourier-mode) likelihood (no compression)
  - 3×2pt-like binned power-spectrum likelihood (compressed)
- Shared plotting utilities and side-by-side constraints.

See the Synthetic demo page for details and quick commands.
