# Contributing Guidelines

- Feature branches (e.g., `feature/desi-ingest`, `feature/fli-likelihood`) are
  expected for development work.
- Pull requests should remain focused and include tests/docs for the scientific
  changes introduced.
- The formatting and test suite should run (`pre-commit run --all-files`,
  `pytest`) before pushing, particularly for modules connected to
  DESI/Planck/ACT workflows.

## Versioning
- The version in `pyproject.toml` should be bumped whenever a release advances
  the DESI × CMB lensing analysis, and 'CITATION.cff' version and release date should be updated accordingly.
