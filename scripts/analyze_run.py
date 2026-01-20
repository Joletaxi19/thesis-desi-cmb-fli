#!/usr/bin/env python
"""
Analyze run results with custom filtering.
Can be used as a CLI script or imported as a module.

Usage:
    python scripts/analyze_run.py --run_dir <RUN_DIR> [OPTIONS]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.chains import Chains

# Enable x64 (needed for some operations)
jax.config.update("jax_enable_x64", True)

def load_and_process_run(run_dir, burn_in=0.5, exclude_chains=None):
    """
    Load samples from a run directory, apply burn-in and chain exclusion,
    and reparameterize to physical space.

    Args:
        run_dir (Path or str): Path to the run directory.
        burn_in (float): Fraction of samples to discard (0.0-1.0).
        exclude_chains (list[int]): Indices of chains to exclude.

    Returns:
        dict: A dictionary containing:
            - physical_samples (dict): Reparameterized physical samples (numpy arrays).
            - physical_chains (Chains): The Chains object (for plotting).
            - truth_vals (dict): Truth values from config.
            - available_params (list): List of available physical parameter names.
            - n_eff (dict): Effective sample size per parameter (if computed).
            - r_hat (dict): R-hat per parameter (if computed).
    """
    run_dir = Path(run_dir)
    config_dir = run_dir / "config"
    if exclude_chains is None:
        exclude_chains = []

    print(f"Loading run: {run_dir}")
    print(f"Burn-in: {burn_in:.0%}")
    if exclude_chains:
        print(f"Excluding chains: {exclude_chains}")

    # 1. Load Model Config
    model_config_path = config_dir / "model.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Could not find model.yaml at {model_config_path}")

    # 1. Load Model Config
    from desi_cmb_fli.model import get_model_from_config

    config_yaml_path = config_dir / "config.yaml"
    print("Instantiating FieldLevelModel from config...")
    model, cfg = get_model_from_config(config_yaml_path) # Loads config internally

    # 2. Load Samples & Truth
    samples_path = config_dir / "samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples_raw = jnp.load(samples_path)

    # 3. Apply Burn-in and Chain Exclusion
    samples_burned = {}

    # Determine chains to keep
    first_key = samples_raw.files[0]
    n_chains_total = samples_raw[first_key].shape[0]
    keep_indices = [i for i in range(n_chains_total) if i not in exclude_chains]

    if len(keep_indices) == 0:
        raise ValueError("All chains excluded!")

    for k in samples_raw.files:
        data = samples_raw[k] # (n_chains, n_samples)
        # 1. Exclude chains
        data_kept = data[keep_indices, :]

        # 2. Burn-in
        _, n_total = data_kept.shape
        n_keep = int(n_total * (1 - burn_in))
        start_idx = n_total - n_keep

        # Handle case where n_keep is 0
        if n_keep <= 0:
            raise ValueError(f"Burn-in {burn_in} is too high, 0 samples kept!")

        samples_burned[k] = jnp.array(data_kept[:, start_idx:])

    print(f"Original chains: {n_chains_total} -> Kept: {len(keep_indices)} (indices {keep_indices})")
    print(f"Samples kept per chain: {n_keep} (discarded first {start_idx})")

    # 4. Reparameterize
    print("Reparametrizing to physical space...")
    # Filter for scalar latent parameters
    samples_jax = {
        k: v for k, v in samples_burned.items()
        if k in model.loc_fid or k.endswith('_')
    }

    chain_obj = Chains(samples_jax, model.groups | model.groups_)
    physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
    physical_samples = {k: np.array(v) for k, v in physical_chains.data.items()}

    # Load Truth values
    config_yaml_path = config_dir / "config.yaml"
    truth_vals = {}
    if config_yaml_path.exists():
        full_cfg = utils.yload(config_yaml_path)
        if "truth_params" in full_cfg:
            truth_vals = full_cfg["truth_params"]

    # Identify available parameters
    comp_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
    available_params = [p for p in comp_params if p in physical_samples]

    return {
        "physical_samples": physical_samples,
        "physical_chains": physical_chains,
        "truth_vals": truth_vals,
        "available_params": available_params,
        "n_kept": n_keep,
        "model": model,  # Return model for validation plots
        "model_config": cfg
    }

def analyze_run(run_dir, burn_in=0.5, exclude_chains=None, output_subdir=None):
    """
    Perform full analysis: load, diagnostics, plots.

    Args:
        run_dir (Path or str): Run directory.
        burn_in (float): Burn-in fraction.
        exclude_chains (list[int]): Chains to exclude.
        output_subdir (str, optional): Name of output directory. If None, generated automatically.
    """
    run_dir = Path(run_dir)

    # Generate default output directory name
    if output_subdir is None:
        suffix_parts = [f"burn{int(burn_in*100)}"]
        if exclude_chains:
            excl_str = "_".join(map(str, exclude_chains))
            suffix_parts.append(f"excl{excl_str}")
        else:
            suffix_parts.append("allchains")
        output_subdir = f"reanalysis_{'_'.join(suffix_parts)}"

    fig_dir = run_dir / output_subdir
    fig_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {fig_dir}")

    # Load data
    data = load_and_process_run(run_dir, burn_in, exclude_chains)
    physical_chains = data["physical_chains"]
    physical_samples = data["physical_samples"]
    truth_vals = data["truth_vals"]
    available_params = data["available_params"]
    model = data.get("model")

    # 5. Diagnostics & Plotting
    print("\n" + "="*40)
    print("DIAGNOSTICS (R-hat & ESS)")
    print("="*40)

    if hasattr(physical_chains, "print_summary"):
        physical_chains.print_summary()
    else:
        print("print_summary method not found on Chains object.")

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

    # Calculate mean parameters for posterior check (Posterior Mean)
    mean_params = {}

    for i, p in enumerate(available_params):
        ax = axes[i]
        vals = physical_samples[p].reshape(-1)
        mean_val = float(np.mean(vals))
        mean_params[p] = mean_val

        ax.hist(vals, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")

        if p in truth_vals:
            ax.axvline(truth_vals[p], color="red", ls="--", lw=2, label="Truth")
        ax.axvline(mean_val, color="green", ls="-", lw=2, label="Mean")

        ax.set_title(p)
        ax.legend()

    # Hide unused subplots
    for j in range(len(available_params), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(fig_dir / "posteriors.png", dpi=150)
    plt.close()

    # Posterior Predictive Check (Spectra)
    if model is not None:
        print("\n" + "="*40)
        print("POSTERIOR PREDICTIVE CHECK (Spectra)")
        print("="*40)
        from desi_cmb_fli.validation import compute_and_plot_spectra

        # Use mean parameters from posterior
        print("Using posterior mean parameters:", mean_params)

        # Prepare parameters: start with truth values, override with posterior means for sampled params
        check_params = truth_vals.copy()
        check_params.update(mean_params)

        try:
             compute_and_plot_spectra(
                model=model,
                truth_params=check_params,
                output_dir=fig_dir,
                n_realizations=1,
                seed=42, # Check a single realization with mean params
                show=False,
                suffix="_posterior_check",
                model_config=data.get("model_config")
             )
        except Exception as e:
             print(f"Warning: Failed to run posterior predictive check: {e}")

    # GetDist
    try:
        from getdist import plots
        print("Generating Triangle Plot...")
        samples_gd = physical_chains.to_getdist()

        g = plots.get_subplot_plotter()
        g.triangle_plot([samples_gd], filled=True, title_limit=1, markers=truth_vals)
        g.export(str(fig_dir / "corner.png"))
        print(f"✓ Analysis complete. Figures in {fig_dir}")

    except ImportError:
        print("GetDist not installed, skipping corner plot.")

def main():
    parser = argparse.ArgumentParser(description="Re-analyze run results with custom filtering.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Path to the run directory")
    parser.add_argument("--burn_in", type=float, default=0.5, help="Fraction of samples to discard (0.0-1.0)")
    parser.add_argument("--exclude_chains", type=int, nargs="*", default=[], help="Indices of chains to exclude (0-indexed)")
    args = parser.parse_args()

    analyze_run(args.run_dir, args.burn_in, args.exclude_chains)

if __name__ == "__main__":
    main()
