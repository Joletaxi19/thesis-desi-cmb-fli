# thesis-desi-cmb-fli

Repository for a thesis project targeting **field-level inference (FLI)** on DESI galaxy samples cross-correlated with CMB lensing reconstructions from Planck and ACT.

## Thesis vision
- Construction of a reproducible FLI workflow that ingests DESI galaxy
  clustering data and CMB lensing maps (Planck PR4, ACT DR6).
- Delivery of joint cosmological constraints from the cross-correlation of those
  observables, with clear validation and documentation.

## Quick Start

**Local development** (CPU only):
```bash
conda env create -f env/environment.yml
conda activate desi-cmb-fli
pip install -e .
pre-commit install
```

**NERSC Perlmutter** (with GPU): See `docs/hpc.md` for complete setup.

## Development

**Run tests**: `pytest`
**Format code**: `ruff format .`
**Preview docs**: `mkdocs serve`

Git hooks automatically format code on commit. CI runs tests on push.

## Pipeline Status

**✅ Completed:** Initial conditions, gravitational evolution, galaxy bias modeling, and field-level inference on synthetic galaxy data

**🚧 Next Steps:**
- CMB lensing modeling (convergence κ from matter fields)
- Field-level inference on synthetic galaxy + CMB lensing data
- Field-level inference on real data (DESI LRG × Planck/ACT κ-maps)

See [`docs/pipeline.md`](docs/pipeline.md) for detailed implementation roadmap and [`notebooks/`](notebooks/) for interactive demonstrations.

### Repository Structure
- `src/desi_cmb_fli/`: Simulation and inference modules
- `configs/`: Experiment configurations (tracer selections, masks, inference grids)
- `tests/`: Unit, integration, and regression tests
- `notebooks/`: Interactive demonstrations and tutorials
- `docs/`: Scientific documentation and pipeline details

## Citation
The project scaffold for the FLI × DESI × CMB lensing analysis should be cited
using `CITATION.cff`.

## License
MIT (see `LICENSE`).
