# thesis-desi-cmb-fli

Repository scaffold for a thesis project targeting **field-level inference (FLI)** on
DESI galaxy samples cross-correlated with CMB lensing reconstructions from
Planck and ACT. The scientific pipeline itself remains unimplemented; the
supporting infrastructure has been preserved so the cosmological workflow can be
rebuilt cleanly.

## Thesis vision
- Construction of a reproducible FLI workflow that ingests DESI galaxy
  clustering data and CMB lensing maps (Planck PR4, ACT DR6+).
- Delivery of joint cosmological constraints from the cross-correlation of those
  observables, with clear validation and documentation.
- Maintenance of publication-grade software practices (tests, docs, metadata)
  throughout thesis development.

## What is already in place
- Python package stub (`desi_cmb_fli`) exposing version metadata.
- Automation hooks: MkDocs-based documentation, pytest configuration, formatting
  via `ruff`/`black`, and a `sync_citation` helper script.
- Repository structure for configs (`configs/`), generated figures (`figures/`),
  and documentation (`docs/`), ready to host the real pipeline components.
- GitHub Actions for continuous testing and doc deployment—pushes trigger the
  test suite and documentation builds rerun automatically when relevant content
  changes.

## Getting started

### Initial setup (one-time)
1. **Create and activate environment**:
   ```bash
   conda env create -f env/environment.yml
   conda activate desi-cmb-fli
   ```

2. **Install package with development tools**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Enable automation hooks** (critical!):
   ```bash
   pre-commit install
   ```

4. **Optional: Install documentation dependencies**:
   ```bash
   pip install -e ".[docs]"
   ```

### Development workflow
After initial setup, the workflow is fully automated. **It is recommended to use feature branches for development**:
```bash
# Create and switch to a new branch for your feature or fix
git checkout -b my-feature-branch

# Write your code...
git add .
git commit -m "Your changes"  # ← Auto-formatting & validation
git push origin my-feature-branch  # ← Auto-testing & doc deployment
```

Once your work is ready, open a pull request to merge your branch into `main`.

### Manual commands (optional)
- Run tests locally: `pytest`
- Check code style: `ruff check .`
- Preview documentation: `mkdocs serve`

### What happens automatically
- **On commit**: Code formatting (black), linting (ruff), citation sync, file validation
- **On push**: Full test suite via GitHub Actions, documentation build/deploy

## Building the cosmology analysis
- DESI/Planck/ACT interfaces and FLI likelihood modules are intended to reside
  under `src/desi_cmb_fli/`.
- Experiment configurations (tracer selections, mask definitions, inference
  grids) should be tracked in `configs/` with lightweight examples committed to
  the repository.
- The `tests/` directory is expected to grow with unit, integration, and
  regression checks covering the forward model, likelihood evaluations, and
  validation diagnostics.
- Scientific logic should be documented in `docs/`, including dataset handling,
  field modeling choices, validation strategies, and interpretation of
  cosmological posteriors.

## Citation
The project scaffold for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
