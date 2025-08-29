# thesis-desi-cmb-fli

Public research code and reproducible pipeline for my PhD project:
**Field-level inference (FLI) for the joint analysis of DESI galaxy clustering and CMB lensing** (Planck PR4, later ACT).

📚 **[Complete Documentation](https://joletaxi19.github.io/thesis-desi-cmb-fli/)** | 🚀 **[Getting Started](https://joletaxi19.github.io/thesis-desi-cmb-fli/getting_started/)** | 📊 **[Pipeline Overview](https://joletaxi19.github.io/thesis-desi-cmb-fli/pipeline/)**

## Goals
- Clean, modular codebase to go from raw inputs to the **final plots** used in the paper(s).
- Reproducibility with CI checks, environment files, and data versioning stubs. Lockfiles will be added as the project stabilizes.
- Open to contributions (issues, PRs).

## Quick start
```bash
# 1) Create environment
conda env create -f env/environment.yml
conda activate desi-cmb-fli

# 2) Install the package (with dev tools)
pip install -e ".[dev]"

# 3) Run smoke tests
pytest -q

# 4) Try the minimal demo (synthetic box → FLI toy)
python scripts/run_demo.py --seed 0 --output figures/toy_summary.png
```

The script accepts `--seed` and `--output` arguments to control the random seed and
where the summary plot is written.

> **Note**: Large datasets and maps are **not** stored in this repo. See the [data access documentation](https://joletaxi19.github.io/thesis-desi-cmb-fli/data_access/) for instructions.

## Documentation

📖 **Full documentation available at: https://joletaxi19.github.io/thesis-desi-cmb-fli/**

Key sections:
- 🚀 [Getting Started](https://joletaxi19.github.io/thesis-desi-cmb-fli/getting_started/) - Installation and first steps
- 📊 [Pipeline Overview](https://joletaxi19.github.io/thesis-desi-cmb-fli/pipeline/) - FLI workflow and stages
- 💾 [Data Access](https://joletaxi19.github.io/thesis-desi-cmb-fli/data_access/) - DESI catalogs and Planck maps
- 🤝 [Contributing](https://joletaxi19.github.io/thesis-desi-cmb-fli/contributing/) - Development guidelines

## Citation
Please cite this repository using `CITATION.cff`.

## License
MIT (see `LICENSE`).
