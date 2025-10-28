# DESI × CMB Lensing Field-Level Inference
## Pipeline Blueprint & Implementation Roadmap

The scientific objective is to extract cosmological information using field-level inference on DESI galaxy clustering jointly with CMB lensing convergence maps from Planck and ACT. The pipeline proceeds through the following stages:

---

## 1. Initial Conditions ✅ (October 2025)

Generate 3D Gaussian random fields representing primordial density fluctuations in a periodic box. These serve as initial conditions for gravitational evolution.

**Module:** `desi_cmb_fli.initial_conditions`
**Implementation:** Power spectrum computation via `jax_cosmo` (Eisenstein–Hu transfer function); 3D field generation in Fourier space; validation through measured power spectra.
**Tests:** `tests/test_initial_conditions.py`
**Notebook:** `notebooks/01_initial_conditions_demo.ipynb`

**Next step:** Gravitational evolution using `jaxpm` to evolve δ(x, z_init) → δ(x, z_obs)

---

## 2. Gravitational Evolution (Next)

Implement particle–mesh (PM) evolution with `jaxpm` to evolve the initial matter field.
Include survey ingestion for DESI tracer selections and Planck/ACT lensing maps, masks, and noise properties.

**Resources:**
- [jaxpm GitHub](https://github.com/DifferentiableUniverseInitiative/jaxpm)

---

## 3. Galaxy Bias Modeling (Planned)

Populate the evolved matter field with galaxies according to bias models.

---

## 4. Lensing Observables (Planned)

Compute convergence fields from the matter distribution and compare to CMB lensing reconstructions.

---

## 5. Likelihood & Inference (Planned)

Develop Fourier-space field-level likelihood and complementary 3×2pt analyses for validation.
Sampling performed with NumPyro (NUTS/HMC).
Design matrix structures, FFT conventions, and marginalization strategies will be specified.

---

## 6. Validation (Planned)

Perform null tests, generate mocks, and run consistency checks spanning instrument systematics to cosmological parameters.
Include cross-comparisons with compressed statistics (e.g., 3×2pt power spectra).

---

## 7. Reporting (Planned)

Produce reproducible figures, tables, and documentation to support thesis deliverables and publications.

---

## NERSC Infrastructure ✅ (October 2025)

- Environment recipe: `env/environment.yml` with JAX, NumPyro, jaxpm, jax_cosmo
- Perlmutter SLURM template: `configs/nersc/slurm/template_perlmutter.sbatch`
- JAX+CUDA 12.4 configuration documented in `docs/hpc.md`
- Storage defined for DESI inputs and scratch outputs
- Configuration files under `configs/nersc/` describe queue, storage, and environment setup

---

Implementation notes, lessons learned, and validation outcomes should continue to be logged here to keep the thesis narrative synchronized with the evolving codebase.
