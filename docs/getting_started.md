# Getting started

The following steps prepare the DESI × CMB lensing FLI scaffold for development:

1. A conda environment can be created via `conda env create -f
   env/environment.yml && conda activate desi-cmb-fli`.
2. Install the package in editable mode with development extras using
   `pip install -e ".[dev]"`.
3. Execute the test suite and static checks through `pytest` and `ruff check .`.
4. Serve the documentation locally with `mkdocs serve` while iterating on the
   analysis notes.

Subsequent work should introduce ingestion, simulation, and inference modules
for DESI galaxies and Planck/ACT CMB lensing maps.
