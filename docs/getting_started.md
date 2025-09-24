# Getting started

Follow these steps to work with the repository skeleton:

1. Create the development environment (for example with conda):
   `conda env create -f env/environment.yml && conda activate desi-cmb-fli`
2. Install the package in editable mode with development extras:
   `pip install -e ".[dev]"`
3. Run the test suite and static checks: `pytest` and `ruff check .`
4. Serve the documentation locally with `mkdocs serve`.

From here you can begin adding your own scientific modules, configs, and docs.
