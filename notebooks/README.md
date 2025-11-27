# Notebooks

Interactive demonstrations and tutorials for the DESI × CMB lensing field-level inference pipeline.

## Running Notebooks

**Interactive**: Open `.ipynb` files in Jupyter/VS Code for exploration and visualization.

**Batch (NERSC Perlmutter)**: For GPU-intensive tasks (MCMC inference), notebooks have been converted to Python scripts in `scripts/` with YAML configuration in `configs/`.
- Scripts support multi-GPU parallelization and use MCLMC for efficiency.
- Outputs (plots, config) are stored in `outputs/run_{timestamp}/`.
- Terminal output is logged in `logs/`.
- See [`docs/hpc.md`](../docs/hpc.md) for detailed instructions.

## Available Notebooks

### 01_initial_conditions_demo.ipynb
**Status:** ✅ Complete
**Purpose:** Demonstrate Gaussian random field generation for initial conditions

**Contents:**
- Power spectrum computation using `jax_cosmo`
- 3D density field generation in a periodic box
- Visualization: 2D slices and density histograms
- Validation: P(k) recovery from generated fields

---

### 02_gravitational_evolution_demo.ipynb
**Status:** ✅ Complete
**Purpose:** Demonstrate gravitational evolution of initial density fields

**Contents:**
- Linear growth factor and growth rate computations
- Lagrangian Perturbation Theory (1LPT and 2LPT)
- N-body evolution with BullFrog solver
- Visualization: evolved density fields and distributions
- Method comparison: 1LPT vs 2LPT vs N-body

---

### 03_galaxy_bias_demo.ipynb
**Status:** ✅ Complete
**Purpose:** Transform matter fields into galaxy fields using bias models

**Contents:**
- Kaiser model (linear bias + RSD in Fourier space)
- Lagrangian bias expansion (LBE) with higher-order terms
- Redshift-space distortions (RSD) from peculiar velocities
- Observational noise modeling (shot noise)
- Power spectrum comparison and validation
- Visual comparison: matter vs galaxy fields

---

### 04_field_level_inference.ipynb
**Status:** ✅ Complete
**Purpose:** Full Bayesian inference pipeline on synthetic data

**Contents:**
- `FieldLevelModel` creation and configuration
- Synthetic observation generation from truth parameters (Gaussian likelihood)
- MCMC inference with NUTS sampler
- Parameter recovery validation
- Visualizations: trace plots, posterior distributions, corner plots

**Note:** To run this analysis (field-level inference without CMB lensing), use `scripts/05_cmb_lensing.py` with `cmb_enabled: false` in the configuration. See "Running Notebooks" above for details.

---


### 05_cmb_lensing.ipynb
**Status:** ✅ Complete
**Purpose:** Demonstrate joint field-level inference with Galaxies + CMB Lensing on CPU

**Contents:**
- Model configuration with CMB lensing enabled
- Synthetic observation generation (Galaxies + CMB convergence $\kappa$)
- Joint inference using MCLMC
- Visualization of results

**Note:** For production runs, use the optimized script `scripts/05_cmb_lensing.py`. See "Running Notebooks" above for details.

---

## Next Notebooks (Planned)

Future notebooks will cover:
- Real data inference with DESI LRG × Planck/ACT κ-maps

Details will be added as implementation progresses.
