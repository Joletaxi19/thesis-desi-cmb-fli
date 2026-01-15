#!/usr/bin/env python
"""
Compare two runs of run_inference.py using GetDist.
Generates a triangle plot comparing the posterior distributions of parameters.

Usage:
    python scripts/compare_runs.py <run_dir1> <run_dir2> [--labels "Label 1" "Label 2"] [--output comparison.png]
"""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from getdist import plots

from desi_cmb_fli.chains import Chains
from desi_cmb_fli.model import FieldLevelModel

# Memory optimization for JAX
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)


def load_and_process(run_dir, label):
    """Load samples from a run directory and process them for GetDist."""
    run_path = Path(run_dir)
    config_dir = run_path / "config"

    print(f"\nProcessing {label}: {run_dir}")

    from desi_cmb_fli import utils

    # Load config
    cfg = utils.yload(config_dir / "config.yaml")

    # Load samples (scalars)
    samples_path = config_dir / "samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples not found at {samples_path}")

    samples_data = np.load(samples_path)
    samples = {k: samples_data[k] for k in samples_data.files}

    # Load truth
    truth_params = cfg["truth_params"]

    # Load model config from model.yaml (saved by run script)
    import yaml
    model_config_path = config_dir / "model.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"model.yaml not found at {model_config_path}")

    with open(model_config_path) as f:
        model_kwargs = yaml.safe_load(f)

    # Convert lists to tuples for shape parameters
    if "box_shape" in model_kwargs:
        model_kwargs["box_shape"] = tuple(model_kwargs["box_shape"])
    if "mesh_shape" in model_kwargs:
        model_kwargs["mesh_shape"] = tuple(model_kwargs["mesh_shape"])

    model = FieldLevelModel(**model_kwargs)

    # Reparametrize to physical space
    print("  Reparametrizing samples...")
    large_fields = ["init_mesh_", "gxy_mesh", "matter_mesh", "lpt_pos", "rsd_pos", "nbody_pos", "obs"]
    samples_jax = {
        k: jnp.array(v)
        for k, v in samples.items()
        if k.endswith('_') and k not in large_fields
    }

    chain_obj = Chains(samples_jax, model.groups | model.groups_)
    physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
    physical_samples = {k: np.array(v) for k, v in physical_chains.data.items()}

    # Prepare for GetDist
    comp_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
    available_params = [p for p in comp_params if p in physical_samples]

    # Create MCSamples
    samples_gd = physical_chains.to_getdist(label=label)

    return samples_gd, truth_params, available_params

def main():
    parser = argparse.ArgumentParser(description="Compare two runs using GetDist")
    parser.add_argument("dir1", help="Path to first run directory")
    parser.add_argument("dir2", help="Path to second run directory")
    parser.add_argument("--labels", nargs=2, default=["Run 1", "Run 2"], help="Labels for the runs")
    parser.add_argument("--output", default="figures/comparison_triangle.png", help="Output filename")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process runs
    samples1, truth1, params1 = load_and_process(args.dir1, args.labels[0])
    samples2, truth2, params2 = load_and_process(args.dir2, args.labels[1])

    common_params = [p for p in params1 if p in params2]

    print("\nGenerating comparison plot...")

    # Prepare markers (truth values)
    # Assume truth is same for both runs, use truth1
    markers = {}
    for p in common_params:
        if p in truth1:
            markers[p] = truth1[p]

    g = plots.get_subplot_plotter()
    g.triangle_plot([samples1, samples2], params=common_params, filled=True, title_limit=1,
                    markers=markers)

    # Save
    g.export(args.output)
    print(f"✓ Comparison plot saved to: {args.output}")

if __name__ == "__main__":
    main()
