# Configuration placeholders

This directory is reserved for experiment configurations as the field-level
inference pipeline takes shape. Typical files may describe DESI tracer
selections, mask combinations, noise models for Planck/ACT lensing maps, or
sampler settings for the FLI likelihood. Maintaining representative YAML/TOML
configs under version control helps document reproducible runs.

The `nersc/` subtree contains Perlmutter-specific environment, storage, and
SLURM templates introduced during Phase 0. Edit the placeholders before
submitting jobs on NERSC systems.
