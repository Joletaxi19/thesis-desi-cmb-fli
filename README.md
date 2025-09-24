# thesis-desi-cmb-fli

This repository now ships as a clean, documented skeleton for a thesis-scale
research codebase. All domain-specific implementations have been intentionally
removed so you can rebuild the scientific parts from scratch while keeping the
support tooling in place.

## What stays in the skeleton
- Python package stub (`desi_cmb_fli`) with a versioned entry point and a
  reusable logging helper.
- Automated docs powered by MkDocs Material (see `mkdocs.yml` and `docs/`).
- Testing and lint hooks configured via `pytest`, `ruff`, `black`, and `mypy`.
- Utility scripts such as `scripts/sync_citation.py` to keep metadata in sync.
- Placeholders for configs (`configs/`) and generated assets (`figures/`).

## Getting started
1. Create the development environment (example with conda):
   `conda env create -f env/environment.yml && conda activate desi-cmb-fli`
2. Install the package locally with the dev extras: `pip install -e ".[dev]"`
3. Run the test suite: `pytest`
4. Serve the docs locally: `mkdocs serve`

## Rebuilding your analysis
- Re-introduce your scientific modules under `src/desi_cmb_fli/`.
- Add configuration files to `configs/` and point your scripts/tests at them.
- Document new components in `docs/` and update `mkdocs.yml` navigation.
- Extend `tests/` with realistic checks for the functionality you add.

## Citation
Please cite this repository using `CITATION.cff` if you build upon the skeleton.

## License
MIT (see `LICENSE`).
