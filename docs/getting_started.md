# Getting started

The following steps prepare the DESI × CMB lensing FLI scaffold for development.

## Complete setup guide

### 1. Environment setup
Create and activate the conda environment:
```bash
cd thesis-desi-cmb-fli
conda env create -f env/environment.yml
conda activate desi-cmb-fli
```

### 2. Package installation
Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

### 3. Enable automation
Activate pre-commit hooks for automatic code formatting and validation:
```bash
pre-commit install
```

### 4. Optional: Documentation tools
For local documentation preview:
```bash
pip install -e ".[docs]"
mkdocs serve  # View at http://localhost:8000
```

## Development workflow

After initial setup, development is fully automated:

1. **Write code** in `src/desi_cmb_fli/`
2. **Commit changes**: `git add . && git commit -m "Description"`
   - Pre-commit hooks automatically format code
3. **Push changes**: `git push`
   - GitHub Actions automatically run tests and deploy documentation

## Validation commands

These commands are optional (automation handles them):
- **Run tests**: `pytest`
- **Check and format code**: `ruff check . && ruff format .`
- **Validate setup**: Try importing the package in Python

## What's automated

- **Pre-commit hooks**: Code formatting (ruff), YAML validation
- **GitHub Actions**: Full test suite, documentation build/deploy on push

After completing the initial setup, you only need to code and push!

## Next steps

Subsequent work should introduce ingestion, simulation, and inference modules
for DESI galaxies and Planck/ACT CMB lensing maps.
