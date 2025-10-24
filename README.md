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
  via `ruff`.
- Repository structure for configs (`configs/`), generated figures (`figures/`),
  and documentation (`docs/`), ready to host the real pipeline components.
- GitHub Actions for continuous testing and doc deployment—pushes trigger the
  test suite and documentation builds rerun automatically when relevant content
  changes.

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

**Run tests**: `pytest` (DESI tests auto-skip on non-NERSC systems)
**Format code**: `ruff format .`
**Preview docs**: `mkdocs serve`

Git hooks automatically format code on commit. CI runs tests on push.

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
