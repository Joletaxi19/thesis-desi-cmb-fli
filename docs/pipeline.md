# Pipeline blueprint

The scientific objective is to extract cosmological information using
field-level inference on DESI galaxy clustering jointly with CMB lensing
convergence maps from Planck and ACT. The future implementation is expected to
cover the following stages:

- **Survey ingestion**: downloading, validating, and documenting DESI tracer
  selections alongside Planck/ACT lensing maps, masks, and noise properties.
- **Forward modeling**: simulating the coupled matter field, galaxy observables,
  and lensing reconstructions needed for FLI-based likelihoods.
- **Likelihood & inference**: developing the Fourier-space field-level
  likelihood together with control analyses (e.g., 3×2pt) for cross-validation.
- **Validation**: performing null tests, generating mocks, and executing
  consistency checks that span instrument systematics to cosmological
  parameters.
- **Reporting**: producing reproducible figures, tables, and documentation to
  support thesis deliverables and eventual publications.

The current repository provides only the scaffolding (package layout, tests,
MkDocs, CI). Each stage should be populated under `src/desi_cmb_fli/` as the
thesis work advances.
