# Repository Review Report

## Scope
- Reviewed all files in the repository (excluding `.git` and `.github/agents`), including
  `scripts/run_inference.py`, `src/desi_cmb_fli/model.py`, `src/desi_cmb_fli/cmb_lensing.py`,
  `src/desi_cmb_fli/validation.py`, plus notebook files, additional scripts, and test files.

## Confirmed issues (blocking)
- None found.

## Potential issues (not confirmed)
- `src/desi_cmb_fli/metrics.py` (`spectrum`): when `mesh2` is provided, the complex cross-spectrum
  bins are combined as `sqrt(real^2 + imag^2)` rather than keeping the real part of the summed
  cross-power. This can bias cross-spectra/coherence to be strictly positive and hide sign changes.
  If a signed cross-spectrum is expected, this should likely be `psum_real + 1j*psum_imag`
  (or just `psum_real` if symmetry makes the imaginary part vanish).
- `src/desi_cmb_fli/model.py` (`FieldLevelModel.__post_init__`): the input `a_obs` value is
  overwritten based on the box center and fiducial cosmology. If a user expects to control
  `a_obs` from config, this behavior could be surprising and lead to mismatched redshift targets.
- `src/desi_cmb_fli/cmb_lensing.py` (`density_field_to_convergence`): when `box_center_chi` is
  smaller than half the box depth, `r_start` becomes negative, leading to negative `r_planes`.
  This can yield invalid `a_of_chi` calls or unphysical lensing kernels for those planes.
- `src/desi_cmb_fli/validation.py` (`compute_and_plot_spectra`): `gxy_delta = (gxy_proj - mean)/mean`
  assumes `mean` is nonzero; if the projection happens to be mean-zero (or negative), this will
  divide by zero or flip signs in unexpected ways.
- `src/desi_cmb_fli/utils.py` (`yload`): uses `yaml.load(..., Loader=yaml.Loader)` which allows
  arbitrary object construction. If untrusted YAML is ever loaded, this is a potential security risk.

## Test Results
- `pytest`: not run successfully (`command not found` in the current environment; README/dev dependencies indicate it should be available after environment setup).
- `ruff check .`: not run successfully (`command not found` in the current environment; pre-commit config uses ruff; README mentions `ruff format .` for formatting).
- Environment setup guidance (README): `conda env create -f env/environment.yml` then `pip install -e .`.
