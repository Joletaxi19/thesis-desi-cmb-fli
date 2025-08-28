# thesis-desi-cmb-fli

Public research code and reproducible pipeline for my PhD project:
**Field-level inference (FLI) for the joint analysis of DESI galaxy clustering and CMB lensing** (Planck PR4, later ACT).

## Goals
- Clean, modular codebase to go from raw inputs to the **final plots** used in the paper(s).
- Full reproducibility with CI checks, pinned environments, and data versioning stubs.
- Open to contributions (issues, PRs).

## Quick start
```bash
# 1) Create environment
conda env create -f env/environment.yml
conda activate desi-cmb-fli

# 2) Install the package in editable mode
pip install -e .

# 3) Run smoke tests
pytest -q

# 4) Try the minimal demo (synthetic box → FLI toy)
python scripts/run_demo.py
```

> **Note**: Large datasets and maps are **not** stored in this repo. See `data/README.md` and `docs/data_access.md` for instructions.

## Structure
```
thesis-desi-cmb-fli/
├─ src/desi_cmb_fli/          # Python package
│  ├─ data/                   # Loaders, I/O, mocks
│  ├─ sim/                    # Differentiable simulators (JAX-PM wrappers, etc.)
│  ├─ inference/              # Inference loops, likelihoods, priors
│  ├─ analysis/               # Power spectra, summaries, plotting
│  ├─ utils/                  # Common helpers, logging, config
│  └─ __init__.py
├─ configs/                   # YAML configs for runs
├─ scripts/                   # CLI entry points
├─ notebooks/                 # Exploration, figures for papers
├─ tests/                     # Unit tests
├─ docs/                      # MkDocs documentation
├─ env/                       # Conda envs & requirements
├─ .github/workflows/         # CI
└─ LICENSE, CITATION.cff, etc.
```

## Citation
Please cite this repository using `CITATION.cff`.

## License
MIT (see `LICENSE`).
