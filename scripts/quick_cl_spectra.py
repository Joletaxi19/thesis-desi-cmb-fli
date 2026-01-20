#!/usr/bin/env python
"""
Quick C_l Spectra Diagnostic Script

Generate C_l power spectra (kappa-kappa, galaxy-kappa, galaxy-galaxy) from
a simulation using the same config.yaml as run_inference.py.

This allows fast visual verification of spectra without running full inference.

Usage:
    python scripts/quick_cl_spectra.py --config configs/inference/config.yaml --cell_size 5.0 --n_realizations 20
"""

import argparse
import os

import jax

from desi_cmb_fli import utils
from desi_cmb_fli.model import get_model_from_config
from desi_cmb_fli.validation import compute_and_plot_spectra

# Memory optimization for JAX on GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick C_l spectra diagnostic from simulated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="configs/inference/config.yaml",
        help="Path to config.yaml (same format as run_inference.py)"
    )
    parser.add_argument(
        "--cell_size", type=float, default=None,
        help="Override cell size in Mpc/h (for testing higher resolution)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (subsequent realizations use seed+1, seed+2, ...)"
    )
    parser.add_argument(
        "--n_realizations", type=int, default=1,
        help="Number of realizations to average over (for error estimation)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: figures)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("QUICK C_l SPECTRA DIAGNOSTIC")
    print("=" * 80)
    print(f"\nJAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Load Config
    if args.cell_size is not None:
        cfg = utils.yload(args.config)
        cfg["model"]["cell_size"] = args.cell_size
    else:
        cfg = args.config

    # Model (factory handles loading)
    model, model_config = get_model_from_config(cfg)

    # Truth parameters
    # Truth parameters
    if isinstance(cfg, str):
         cfg_dict = utils.yload(cfg)
    else:
         cfg_dict = cfg

    truth_params = cfg_dict["truth_params"]

    # Run Validation
    compute_and_plot_spectra(
        model=model,
        truth_params=truth_params,
        output_dir=args.output_dir,
        n_realizations=args.n_realizations,
        seed=args.seed if args.seed is not None else cfg_dict.get("seed"),
        show=args.show,
        model_config=model_config
    )

if __name__ == "__main__":
    main()
