# thesis-desi-cmb-fli

Repository for a thesis project targeting **field-level inference (FLI)** on DESI galaxy samples cross-correlated with CMB lensing reconstructions from Planck and ACT.

## Thesis vision
- Construction of a reproducible FLI workflow that ingests DESI galaxy
  clustering data and CMB lensing maps (Planck PR4, ACT DR6).
- Delivery of joint cosmological constraints from the cross-correlation of those
  observables, with clear validation and documentation.

## Quick Start

**Local development** (CPU only):
```bash
conda env create -f env/environment.yml
conda activate desi-cmb-fli
pip install -e .
pre-commit install
```

**NERSC Perlmutter** (with GPU): See `docs/hpc.md` for complete setup.

## Development

**Run tests**: `pytest`
**Format code**: `ruff format .`
**Preview docs**: `mkdocs serve`

Git hooks automatically format code on commit. CI runs tests on push.

## Building the cosmology analysis

### Current Status

**✅ Completed:**
- **Initial Conditions** (`src/desi_cmb_fli/initial_conditions.py`): Gaussian random field generation for cosmological simulations
  - Linear matter power spectrum computation via jax_cosmo
  - 3D density field generation
  - Validation tools for P(k) recovery
  - Demo notebook

- **Gravitational Evolution** (`src/desi_cmb_fli/evolution.py`): Time evolution of density fields
  - Lagrangian Perturbation Theory (1LPT, 2LPT)
  - N-body particle-mesh solver with BullFrog integrator
  - Growth factor computations
  - Demo notebook

- **Galaxy Bias** (`src/desi_cmb_fli/galaxy_bias.py`): Transforming matter to galaxy fields
  - Kaiser model (linear bias + RSD in Fourier space)
  - Lagrangian bias expansion (LBE) for non-linear effects
  - Redshift-space distortions (RSD)
  - Observational noise modeling
  - Demo notebook

**🚧 Next Steps:**
- DESI/Planck/ACT data interfaces
- Field-level likelihood implementation
- MCMC sampling infrastructure

### Repository Structure
- `src/desi_cmb_fli/`: Simulation and inference modules
- `configs/`: Experiment configurations (tracer selections, masks, inference grids)
- `tests/`: Unit, integration, and regression tests
- `notebooks/`: Interactive demonstrations and tutorials
- `docs/`: Scientific documentation and pipeline details

## Citation
The project scaffold for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
