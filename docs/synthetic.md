# Implementation roadmap

This living document records technical plans, open questions, and milestones for
the DESI × CMB lensing field-level inference pipeline.

Suggested checkpoints:

1. **Data interfaces** – wrappers for DESI redshift catalogs, masks, and
   Planck/ACT convergence maps, including metadata and QA summaries.
2. **Simulation prototypes** – minimal forward models (Gaussian or N-body
   inspired) that mirror the expected survey geometry and noise to stress-test
   the inference code.
3. **Field-level likelihood** – design matrix structures, FFT conventions,
   marginalization strategy, and numerical stability considerations.
4. **Validation suite** – mock-based coverage checks, null tests, and
   cross-comparisons with compressed statistics (e.g., 3×2pt).
5. **Result packaging** – figure generation, notebook recipes, and narrative
   documentation of cosmological outcomes.

Implementation notes and lessons learned should be logged here to keep the
thesis narrative synchronized with the codebase.
