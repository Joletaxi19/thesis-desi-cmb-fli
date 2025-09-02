# Getting started

- Create the conda environment: `conda env create -f env/environment.yml && conda activate desi-cmb-fli`
- Install the package with dev tools: `pip install -e ".[dev]"`
- Run tests: `pytest -q`

Quick demo
- Synthetic constraints (FLI vs 3×2pt) via YAML:
  `python scripts/run_synthetic_comparison.py --config configs/demo.yaml`
