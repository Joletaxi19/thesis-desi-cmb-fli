# Notebooks

Interactive demonstrations and tutorials for the DESI × CMB lensing field-level inference pipeline.

## Running Notebooks

**On NERSC Perlmutter**:

### Batch Job (sbatch)

For long-running, GPU-intensive tasks (like MCMC inference), it is highly recommended to run them as non-interactive batch jobs.

Notebooks intended for this (e.g., `04-fli-inference.ipynb`) have been converted into Python scripts and are located in the `scripts/` directory. These script versions are optimized for batch execution and **adapted for parallelism** (e.g., running MCMC chains in parallel across multiple GPUs).

To run a job, submit the sbatch template script, which is pre-configured to execute one of these Python scripts:

```bash
sbatch configs/nersc/slurm/template_perlmutter.sbatch
```

**Note**: Create `logs/` and `output/` directories before submitting to store logs and outputs.

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
- Synthetic observation generation from truth parameters
- MCMC inference with NumPyro NUTS sampler
- Parameter recovery validation
- Visualizations: trace plots, posterior distributions, corner plots

---

## Next Notebooks (Planned)

Future notebooks will cover:
- CMB lensing modeling
- Joint inference on synthetic galaxy + CMB lensing data
- Real data inference with DESI LRG × Planck/ACT κ-maps

Details will be added as implementation progresses.
