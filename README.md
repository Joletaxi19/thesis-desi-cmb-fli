# thesis-desi-cmb-fli

Public research code and reproducible pipeline for my PhD project:
**Field-level inference (FLI) for the joint analysis of DESI galaxy clustering and CMB lensing** (Planck PR4, later ACT).

📚 **[Documentation](https://joletaxi19.github.io/thesis-desi-cmb-fli/)** | 🚀 **[Getting Started](https://joletaxi19.github.io/thesis-desi-cmb-fli/getting_started/)**

## Goals
- Clean, modular codebase to go from raw inputs to the **final plots** used in the paper(s).
- Reproducibility with CI checks, environment files, and data versioning stubs. Lockfiles will be added as the project stabilizes.
- Open to contributions (issues, PRs).

## New: synthetic FLI vs 3×2pt demo
Run a fast, CPU-only comparison of field-level and 3×2pt constraints on the same synthetic maps (YAML-config only):

```
python scripts/run_synthetic_comparison.py --config configs/demo.yaml
```

This generates correlated galaxy and shear maps, scans the likelihood for (A, b), and
plots both posteriors together with 1D marginals. See the docs for details.

> Note: Large datasets and maps are not stored in this repo. See the [Data Access](https://joletaxi19.github.io/thesis-desi-cmb-fli/data_access/) page.

## Citation
Please cite this repository using `CITATION.cff`.

## License
MIT (see `LICENSE`).
