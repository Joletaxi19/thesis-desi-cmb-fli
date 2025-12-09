# DESI × CMB Lensing Field-Level Inference
## Pipeline Blueprint & Implementation Roadmap

The scientific objective is to extract cosmological information using field-level inference on DESI galaxy clustering jointly with CMB lensing convergence maps from Planck and ACT.

## Overview

![Field-Level Inference Architecture](figures/fli_architecture.jpeg)

*Figure: Complete field-level inference pipeline. The forward model (top) generates predictions from initial conditions δ_ini through N-body evolution (JAX-PM) to final density δ_final, which produces both galaxy overdensity δ_g and κ-map through ray-tracing. The MCLMC sampler compares these predictions with observed data (DESI LRG + Planck/ACT κ-map) to update the posterior distribution of cosmological parameters and the initial density field.*

The pipeline proceeds through the following stages:

---

## 1. Initial Conditions ✅ (October 2025)

Generate 3D Gaussian random fields representing primordial density fluctuations in a periodic box. These serve as initial conditions for gravitational evolution.

**Module:** `desi_cmb_fli.initial_conditions`
**Implementation:** Power spectrum computation via `jax_cosmo` (Eisenstein–Hu transfer function); 3D field generation in Fourier space; validation through measured power spectra.
**Tests:** `tests/test_initial_conditions.py`
**Notebook:** `notebooks/01_initial_conditions_demo.ipynb`

**Next step:** Gravitational evolution using `jaxpm` to evolve δ(x, z_init) → δ(x, z_obs)

---

## 2. Gravitational Evolution ✅ (October 2025)

Evolve initial density fields forward in time using Lagrangian Perturbation Theory (LPT) and N-body particle-mesh methods with the BullFrog integrator.

**Module:** `desi_cmb_fli.evolution`
**Implementation:**
- Growth factor computations via `jaxpm` ODE solver
- First and second-order LPT displacements
- Full N-body evolution with BullFrog solver from `diffrax`
- Particle-mesh force calculations

**Tests:** `tests/test_evolution.py`
**Notebook:** `notebooks/02_gravitational_evolution_demo.ipynb`

**Next step:** Galaxy bias modeling and redshift-space distortions

---

## 3. Galaxy Bias Modeling ✅ (October 2025)

Transform evolved matter density fields into galaxy density fields using bias models and redshift-space distortions. Add observational noise to create realistic mock data.

**Modules:** `desi_cmb_fli.bricks`, `desi_cmb_fli.nbody`
**Implementation:**
- Kaiser model: Linear bias + RSD in Fourier space
- Lagrangian bias expansion (LBE): Non-linear bias effects following [Modi+2020](http://arxiv.org/abs/1910.07097)
  - Linear bias (b₁)
  - Quadratic bias (b₂)
  - Tidal shear bias (b_s²)
  - Non-local bias (b_∇²)
- Redshift-space distortions from peculiar velocities
- Observational noise (shot noise) modeling
- Posterior sampling with Kaiser approximation

**Tests:** `tests/test_model.py`
**Notebook:** `notebooks/03_galaxy_bias_demo.ipynb`

**Next step:** Field-level inference pipeline

---

## 4. Field-Level Inference ✅ (October 2025)

Full Bayesian inference pipeline for parameter estimation from synthetic and real data.

**Module:** `desi_cmb_fli.model` (FieldLevelModel)
**Implementation:**
- Probabilistic model:
  - Prior distributions for cosmology (Ωm, σ8) based on Planck18
  - Prior distributions for bias parameters (b₁, b₂, b_s², b_∇²)
  - Prior on initial conditions (Gaussian random field)
- Forward model pipeline: IC → evolution (LPT/Kaiser) → bias → observable
- Data conditioning and model blocking
- Kaiser posterior initialization for efficient MCMC warmup
- NUTS or MCLMC sampler for parameter inference

**Supporting modules:**
- `desi_cmb_fli.metrics` - Analysis metrics and power spectra
- `desi_cmb_fli.chains` - Chain utilities and diagnostics
- `desi_cmb_fli.plot` - Visualization tools
- `desi_cmb_fli.samplers` - MCMC samplers (NUTS, MCLMC)

**Tests:** `tests/test_model.py` (11 tests covering model creation, prediction, conditioning, initialization)
**Notebook:** `notebooks/04_field_level_inference.ipynb` - Complete demonstration with synthetic data

---

## 5. CMB Lensing Modeling ✅ (November 2025)

Computation of convergence κ from matter fields using Born approximation.

**Module:** `desi_cmb_fli.cmb_lensing`
**Implementation:**
- Born approximation integration through lightcone
- Lensing kernel integration
- Projection of density fields to convergence maps

---

> [!IMPORTANT]
> **Original Contribution:** All subsequent sections (Step 6 onwards) represent original developments and contributions implemented specifically for this thesis project.

## 6. Joint Field-Level Inference ✅ (November 2025)

Joint inference on synthetic galaxy + CMB lensing data.

**Script:** `scripts/05_cmb_lensing.py`
**Implementation:**
- Joint likelihood: $P(\delta_g, \kappa | \delta_{ini}, \theta)$
- Constrains cosmology from both tracers
- Validated with cross-correlations
- **Realistic Lensing Noise:**
  - Uses the true Planck lensing noise spectrum ($N_\ell^{\kappa\kappa}$) from PR4.
  - Includes an option to **rescale the noise** to achieve a Signal-to-Noise Ratio (SNR) representative of the true data, accounting for the fact that the simulation covers only a small portion of the line of sight (for validation purposes).
  - This is handled by `compute_signal_capture_ratio` in `cmb_lensing.py`.
- **Automatic Field Size:** The angular size of the field is automatically calculated based on the physical box size and the observation redshift, ensuring the correct geometry for the lensing kernel.
- **Validation:** `scripts/check_box_size_impact.py` allows verifying the impact of box size on the lensing kernel coverage, in function of the redshift of observation.

---

## Next Steps

1. **Inform the model of the missing lensing signal through a noise spectrum that accounts for the missing signal.**

2. **Field-level inference on real data** - Application to DESI LRG × Planck/ACT κ-maps

Implementation details will be documented as development progresses.
