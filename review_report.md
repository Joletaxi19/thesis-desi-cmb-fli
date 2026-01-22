# Repository Review Report

## Scope
- Reviewed all files in the repository (excluding `.git` and `.github/agents`), including
  `scripts/run_inference.py`, `src/desi_cmb_fli/model.py`, `src/desi_cmb_fli/cmb_lensing.py`,
  `src/desi_cmb_fli/validation.py`, plus notebook files, additional scripts, and test files.

## Confirmed issues (blocking)
- None found.

## Potential issues (not confirmed)
- `src/desi_cmb_fli/metrics.py` (`spectrum`): when `mesh2` is provided, cross-power bins are reduced
  to a magnitude `sqrt(real^2 + imag^2)`. This makes cross-spectra strictly positive and can mask
  negative correlations. If you expect a signed cross-spectrum, consider keeping the real
  cross-power (or complex sum) instead of the magnitude.
- `src/desi_cmb_fli/model.py` (`FieldLevelModel.__post_init__`): the input `a_obs` value is
  overwritten based on the box center and fiducial cosmology. If a user expects to control
  `a_obs` from config, this behavior could be surprising and lead to mismatched redshift targets.
- `src/desi_cmb_fli/cmb_lensing.py` (`density_field_to_convergence`): when `box_center_chi` is
  smaller than half the box depth, `r_start` becomes negative, leading to negative `r_planes`.
  This can yield invalid `a_of_chi` calls or unphysical lensing kernels. Consider validating
  inputs or clamping `r_start`/`r_planes` to zero.
- `src/desi_cmb_fli/validation.py` (`compute_and_plot_spectra`): `gxy_delta = (gxy_proj - mean)/mean`
  assumes `mean` is nonzero; if the projection happens to be mean-zero (or negative), this will
  divide by zero or flip signs in unexpected ways.
- `src/desi_cmb_fli/utils.py` (`yload`): uses `yaml.load(..., Loader=yaml.Loader)` which allows
  arbitrary object construction. If untrusted YAML is ever loaded, this is a security risk—consider
  `yaml.safe_load()` as the safer default.

## Test Results
- `pytest`: not run successfully (`command not found` in the current environment; README/dev dependencies indicate it should be available after environment setup).
- `ruff check .`: not run successfully (`command not found` in the current environment; pre-commit config uses ruff; README mentions `ruff format .` for formatting).
- Environment setup guidance (README): `conda env create -f env/environment.yml` then `pip install -e .`.
