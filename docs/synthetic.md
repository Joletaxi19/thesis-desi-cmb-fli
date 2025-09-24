# Extending the skeleton

With the scientific code removed, you can rebuild the repository to match your
analysis needs. A suggested workflow:

1. Sketch the data flow you require (simulation, ingestion, inference, plots).
2. Recreate the package submodules you need under `src/desi_cmb_fli/`.
3. Add configuration files under `configs/` for reproducible runs.
4. Write targeted tests under `tests/` to lock in expected behaviour.
5. Document every major component in this documentation site so future readers
   can follow your pipeline.

Use this page to capture design decisions, prototypes, or comparison studies as
you reintroduce domain logic.
