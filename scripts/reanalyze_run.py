#!/usr/bin/env python
"""
Re-analyze an existing run, allowing for custom burn-in and chain exclusion.

Usage:
    python scripts/reanalyze_run.py --run_dir <RUN_DIR> [OPTIONS]

Options:
    --burn_in <FLOAT>       Fraction of samples to discard (0.0 - 1.0). Default: 0.5
    --exclude_chains <INTS> List of chain indices (0-indexed) to exclude. Default: None

Examples:
    # Basic re-analysis (50% burn-in, keep all chains)
    python scripts/reanalyze_run.py --run_dir outputs/run_20251217_120000

    # Custom burn-in (30%), exclude chain 3 (the 4th one)
    python scripts/reanalyze_run.py --run_dir outputs/run_... --burn_in 0.3 --exclude_chains 3

    # Exclude multiple chains (1 and 3)
    python scripts/reanalyze_run.py --run_dir outputs/run_... --exclude_chains 1 3
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli import utils
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.model import FieldLevelModel

jax.config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser(description="Re-analyze run results with custom filtering.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Path to the run directory")
    parser.add_argument("--burn_in", type=float, default=0.5, help="Fraction of samples to discard (0.0-1.0)")
    parser.add_argument("--exclude_chains", type=int, nargs="*", default=[], help="Indices of chains to exclude (0-indexed)")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_dir = run_dir / "config"

    # Dynamic output folder name based on params
    suffix_parts = [f"burn{int(args.burn_in*100)}"]
    if args.exclude_chains:
        excl_str = "_".join(map(str, args.exclude_chains))
        suffix_parts.append(f"excl{excl_str}")
    else:
        suffix_parts.append("allchains")

    out_folder_name = f"reanalysis_{'_'.join(suffix_parts)}"
    fig_dir = run_dir / out_folder_name
    fig_dir.mkdir(exist_ok=True, parents=True)

    print(f"Re-analyzing run: {run_dir}")
    print(f"Output directory: {fig_dir}")
    print(f"Burn-in fraction: {args.burn_in:.0%}")
    if args.exclude_chains:
        print(f"Excluding chains: {args.exclude_chains}")

    # 1. Load Model Config
    print("Loading model config...")
    model_config_path = config_dir / "model.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Could not find model.yaml at {model_config_path}")

    with open(model_config_path) as f:
       model_kwargs = yaml.safe_load(f)

    if "box_shape" in model_kwargs:
        model_kwargs["box_shape"] = tuple(model_kwargs["box_shape"])
    if "mesh_shape" in model_kwargs:
        model_kwargs["mesh_shape"] = tuple(model_kwargs["mesh_shape"])

    print("Instantiating FieldLevelModel...")
    model = FieldLevelModel(**model_kwargs)

    # 2. Load Samples & Truth
    print("Loading samples and truth...")
    samples_path = config_dir / "samples.npz"

    samples_raw = jnp.load(samples_path)

    # 3. Apply Burn-in and Chain Exclusion
    samples_burned = {}

    # Determine chains to keep
    first_key = samples_raw.files[0]
    n_chains_total = samples_raw[first_key].shape[0]
    keep_indices = [i for i in range(n_chains_total) if i not in args.exclude_chains]

    if len(keep_indices) == 0:
        raise ValueError("All chains excluded!")

    for k in samples_raw.files:
        data = samples_raw[k] # (n_chains, n_samples)
        # 1. Exclude chains
        data_kept = data[keep_indices, :]

        # 2. Burn-in
        n_chains, n_total = data_kept.shape
        n_keep = int(n_total * (1 - args.burn_in))
        start_idx = n_total - n_keep

        # Handle case where n_keep is 0
        if n_keep <= 0:
            raise ValueError(f"Burn-in {args.burn_in} is too high, 0 samples kept!")

        samples_burned[k] = jnp.array(data_kept[:, start_idx:])

    print(f"Original chains: {n_chains_total} -> Kept: {len(keep_indices)} (indices {keep_indices})")
    print(f"Original samples per chain: {n_total}")
    print(f"Kept samples per chain: {n_keep} (discarded first {start_idx})")

    # 4. Reparameterize
    print("\nReparametrizing to physical space...")
    samples_jax = {
        k: v for k, v in samples_burned.items()
        if k in model.loc_fid or k.endswith('_')
    }

    chain_obj = Chains(samples_jax, model.groups | model.groups_)
    physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
    physical_samples = {k: np.array(v) for k, v in physical_chains.data.items()}

    # Truth values
    config_yaml_path = config_dir / "config.yaml"
    truth_vals = {}
    if config_yaml_path.exists():
        full_cfg = utils.yload(config_yaml_path)
        if "truth_params" in full_cfg:
            truth_vals = full_cfg["truth_params"]

    # 5. Diagnostics & Plotting
    print("\n" + "="*40)
    print("DIAGNOSTICS (R-hat & ESS)")
    print("="*40)
    if hasattr(physical_chains, "print_summary"):
        physical_chains.print_summary()
    else:
        print("print_summary method not found on Chains object.")

    comp_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
    available_params = [p for p in comp_params if p in physical_samples]

    # Traces
    print("Plotting Traces...")
    plt.figure(figsize=(10, 2 * len(available_params)))
    physical_chains.plot(names=available_params, grid=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "traces.png", dpi=150)
    plt.close()

    # Posteriors
    print("Plotting Posteriors...")
    rows = (len(available_params) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
    axes = axes.flatten()

    for i, p in enumerate(available_params):
        ax = axes[i]
        vals = physical_samples[p].reshape(-1)
        ax.hist(vals, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")

        if p in truth_vals:
            ax.axvline(truth_vals[p], color="red", ls="--", lw=2, label="Truth")
        ax.axvline(np.mean(vals), color="green", ls="-", lw=2, label="Mean")

        ax.set_title(p)
        ax.legend()

    # Hide unused subplots
    for j in range(len(available_params), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(fig_dir / "posteriors.png", dpi=150)
    plt.close()

    # GetDist
    try:
        from getdist import plots
        print("Generating Triangle Plot...")
        samples_gd = physical_chains.to_getdist()

        g = plots.get_subplot_plotter()
        g.triangle_plot([samples_gd], filled=True, title_limit=1, markers=truth_vals)
        g.export(str(fig_dir / "corner.png"))
        print(f"✓ Re-analysis complete. Figures in {fig_dir}")

    except ImportError:
        print("GetDist not installed.")

if __name__ == "__main__":
    main()
