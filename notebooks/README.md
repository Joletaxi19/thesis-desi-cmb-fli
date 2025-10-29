# Notebooks

Interactive demonstrations and tutorials for the DESI × CMB lensing field-level inference pipeline.

## Available Notebooks

### 01_initial_conditions_demo.ipynb
**Status:** ✅ Complete
**Purpose:** Demonstrate Gaussian random field generation for initial conditions

**Contents:**
- Power spectrum computation using `jax_cosmo`
- 3D density field generation in a periodic box
- Visualization: 2D slices and density histograms
- Validation: P(k) recovery from generated fields

**Usage:**
```bash
# On NERSC Perlmutter (login node, CPU-only)
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
jupyter notebook 01_initial_conditions_demo.ipynb

# For GPU access (interactive job)
salloc --nodes 1 --qos interactive --time 00:10:00 --constraint gpu --gpus 1 --account=desi
jupyter notebook 01_initial_conditions_demo.ipynb
```
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

**Usage:**
```bash
# Same as 01_initial_conditions_demo.ipynb
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
jupyter notebook 02_gravitational_evolution_demo.ipynb
```

---

## Coming Soon

### 03_galaxy_bias.ipynb (Planned)
Populate matter fields with galaxies using bias expansion models.

### 04_likelihood_demo.ipynb (Planned)
Field-level likelihood evaluation and gradient computation.

### 05_inference_nuts.ipynb (Planned)
Parameter inference using NumPyro NUTS sampler.
