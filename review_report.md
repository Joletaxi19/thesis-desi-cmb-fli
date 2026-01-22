# Repository Review Report

## Scope
- Reviewed all files in the repository (excluding `.git` and `.github/agents`), including
  `scripts/run_inference.py`, `src/desi_cmb_fli/model.py`, `src/desi_cmb_fli/cmb_lensing.py`,
  and `src/desi_cmb_fli/validation.py`.

## Issues (100% certain)
- None found.

## Notes on certainty
- Only findings that I could validate with complete certainty are listed above.
- The absence of issues here does not imply that every aspect is perfect; it reflects the
  requested threshold for reporting.

## Test Results
- `pytest`: not run successfully (`command not found` in the current environment; README/dev dependencies indicate it should be available after environment setup).
- `ruff check .`: not run successfully (`command not found` in the current environment; pre-commit config uses ruff; README mentions `ruff format .` for formatting).
- Environment setup guidance (README): `conda env create -f env/environment.yml` then `pip install -e .`.
