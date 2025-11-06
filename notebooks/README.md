# Notebooks

Interactive demonstrations and tutorials for the DESI × CMB lensing field-level inference pipeline.

## Running Notebooks

**Interactive**: Open `.ipynb` files in Jupyter/VS Code for exploration and visualization.

**Batch (NERSC Perlmutter)**: For GPU-intensive tasks (MCMC inference), notebooks have been converted to Python scripts in `scripts/` with YAML configuration in `configs/`.

See [`docs/hpc.md`](../docs/hpc.md) for detailed instructions on running batch jobs.

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

**Note:** The script version (`scripts/04_field_level_inference.py`) is optimized for batch execution on NERSC Perlmutter with multi-GPU parallelization and uses MCLMC instead of NUTS for efficiency. It works with a configuration YAML file, see [`docs/hpc.md`](../docs/hpc.md) for details. The used configuration file and the plots generated are stored in 'outputs/run_{timestamp}/'. The terminal output can be followed in real-time in the 'logs/' directory.

---

## Next Notebooks (Planned)

Future notebooks will cover:
- CMB lensing modeling
- Joint inference on synthetic galaxy + CMB lensing data
- Real data inference with DESI LRG × Planck/ACT κ-maps

Details will be added as implementation progresses.
