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
from datetime import datetime
from pathlib import Path

# Memory optimization for JAX on GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import cmb_lensing, metrics, utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kg,
)
from desi_cmb_fli.model import FieldLevelModel, default_config

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

    # Optional overrides for quick testing at different resolutions
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

    # Load config
    print(f"\n📄 Loading config: {args.config}")
    cfg = utils.yload(args.config)

    # Build model config (same logic as run_inference.py)
    model_config = default_config.copy()
    model_config["box_shape"] = tuple(cfg["model"]["box_shape"])

    # Cell size (with optional override)
    cell_size = args.cell_size if args.cell_size is not None else float(cfg["model"]["cell_size"])
    mesh_shape = [int(round(L / cell_size)) for L in model_config["box_shape"]]
    mesh_shape = [n + 1 if n % 2 != 0 else n for n in mesh_shape]
    model_config["mesh_shape"] = tuple(mesh_shape)

    model_config["evolution"] = cfg["model"]["evolution"]
    model_config["lpt_order"] = cfg["model"]["lpt_order"]
    model_config["gxy_density"] = cfg["model"]["gxy_density"]

    # CMB lensing config
    cmb_cfg = cfg.get("cmb_lensing", {})
    # Force CMB enabled for this diagnostic script
    model_config["cmb_enabled"] = True
    model_config["cmb_field_size_deg"] = None  # Auto-calculate
    model_config["cmb_field_npix"] = int(cmb_cfg["field_npix"]) if "field_npix" in cmb_cfg else None
    model_config["full_los_correction"] = cmb_cfg.get("full_los_correction", False)
    model_config["cmb_z_source"] = float(cmb_cfg.get("z_source", 1100.0))

    if "cmb_noise_nell" in cmb_cfg:
        model_config["cmb_noise_nell"] = cmb_cfg["cmb_noise_nell"]
    else:
        # Fallback: synthetic white noise
        print("⚠️  No cmb_noise_nell in config, using synthetic white noise")
        ell = np.arange(2, 5000)
        nell = np.ones_like(ell) * 5e-8
        model_config["cmb_noise_nell"] = {"ell": ell, "N_ell": nell}

    print(f"\n📦 Box: {model_config['box_shape']} Mpc/h")
    print(f"📐 Mesh: {model_config['mesh_shape']}")
    print(f"📏 Cell size: {cell_size} Mpc/h")
    print(f"🔄 Evolution: {model_config['evolution']}")
    print(f"📡 Full LoS correction: {model_config['full_los_correction']}")

    # Create model
    print("\n" + "-" * 40)
    print("Creating model...")
    model = FieldLevelModel(**model_config)
    print(f"  Field size: {model.cmb_field_size_deg:.2f}°")
    print(f"  CMB npix: {model.cmb_field_npix}")

    # Truth parameters from config
    truth_params = cfg["truth_params"]
    base_seed = args.seed if args.seed is not None else cfg["seed"]
    n_realizations = args.n_realizations

    print("\n" + "-" * 40)
    print("Truth parameters:")
    for k, v in truth_params.items():
        print(f"  {k}: {v}")
    print(f"  base seed: {base_seed}")
    print(f"  n_realizations: {n_realizations}")

    # Get cosmology for projection (same for all realizations)
    cosmo_val = get_cosmology(**truth_params)
    chi_center_val = float(jc.background.radial_comoving_distance(
        cosmo_val, jnp.atleast_1d(model.a_obs)
    )[0])
    field_size = model.cmb_field_size_deg

    # Storage for spectra across realizations
    all_cl_kk_box = []
    all_cl_kk_obs = []
    all_cl_gg = []
    all_cl_kg = []
    ell = None

    # =========================================================================
    # LOOP OVER REALIZATIONS
    # =========================================================================
    print("\n" + "-" * 40)
    print(f"Generating {n_realizations} realization(s)...")

    for i_real in range(n_realizations):
        seed = base_seed + i_real
        print(f"\n  Realization {i_real+1}/{n_realizations} (seed={seed})", end="")

        # Generate truth
        truth = model.predict(
            samples=truth_params,
            hide_base=False,
            hide_samp=False,
            hide_det=False,
            frombase=True,
            rng=jr.key(seed),
        )

        # Project galaxy field
        gxy_field = np.array(truth["obs"])
        gxy_proj = cmb_lensing.project_flat_sky(
            gxy_field,
            model_config["box_shape"],
            model.cmb_field_size_deg,
            model.cmb_field_npix,
            chi_center_val
        )

        # Compute kappa from matter field
        kappa_box = cmb_lensing.density_field_to_convergence(
            truth['matter_mesh'],
            model.box_shape,
            cosmo_val,
            model.cmb_field_size_deg,
            model.cmb_field_npix,
            model.cmb_z_source,
            box_center_chi=chi_center_val,
        )
        if kappa_box.ndim == 3 and kappa_box.shape[0] == 1:
            kappa_box = kappa_box.squeeze(0)
        kappa_box = jnp.asarray(kappa_box, dtype=jnp.float64)

        kappa_obs = truth['kappa_obs']

        # Compute spectra
        ell_i, cl_kk_box_i = metrics.get_cl_2d(kappa_box, field_size_deg=field_size)
        _, cl_kk_obs_i = metrics.get_cl_2d(kappa_obs, field_size_deg=field_size)

        gxy_mean = jnp.mean(gxy_proj)
        gxy_delta = (gxy_proj - gxy_mean) / gxy_mean
        gxy_mean = jnp.mean(gxy_proj)
        gxy_delta = (gxy_proj - gxy_mean) / gxy_mean
        _, cl_gg_i = metrics.get_cl_2d(gxy_delta, field_size_deg=field_size)
        _, cl_kg_i = metrics.get_cl_2d(kappa_box, gxy_delta, field_size_deg=field_size)

        # Store
        if ell is None:
            ell = ell_i
        all_cl_kk_box.append(np.array(cl_kk_box_i))
        all_cl_kk_obs.append(np.array(cl_kk_obs_i))
        all_cl_gg.append(np.array(cl_gg_i))
        all_cl_kg.append(np.array(cl_kg_i))

        print(" ✓")

    # Compute mean and std
    all_cl_kk_box = np.array(all_cl_kk_box)
    all_cl_kk_obs = np.array(all_cl_kk_obs)
    all_cl_gg = np.array(all_cl_gg)
    all_cl_kg = np.array(all_cl_kg)

    cl_kk_box_mean = np.nanmean(all_cl_kk_box, axis=0)
    cl_kk_box_std = np.nanstd(all_cl_kk_box, axis=0)
    cl_kk_obs_mean = np.nanmean(all_cl_kk_obs, axis=0)
    cl_kk_obs_std = np.nanstd(all_cl_kk_obs, axis=0)
    cl_gg_mean = np.nanmean(all_cl_gg, axis=0)
    cl_gg_std = np.nanstd(all_cl_gg, axis=0)
    cl_kg_mean = np.nanmean(all_cl_kg, axis=0)
    cl_kg_std = np.nanstd(all_cl_kg, axis=0)

    print("\n  ✓ All realizations computed")

    # =========================================================================
    # COMPUTE THEORETICAL SPECTRA
    # =========================================================================
    print("  Computing theoretical spectra (Limber)...")

    # Box chi range - use small positive chi_min to avoid division by zero in Limber
    chi_min = 1.0  # Small positive value to avoid div by 0
    chi_max = float(model.box_shape[2])  # Box extends to this chi
    z_source = model.cmb_z_source
    b1 = truth_params.get("b1", 1.0)

    print(f"    chi range: [{chi_min}, {chi_max}] Mpc/h, z_source={z_source}, b1={b1}")

    # Use a clean ell grid for theory (measured ell may have NaN from empty bins)
    ell_clean = np.geomspace(10, 2000, 100)  # Clean log-spaced grid

    cl_kk_theory_clean = compute_theoretical_cl_kappa(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source)
    cl_gg_theory_clean = compute_theoretical_cl_gg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, b1)
    cl_kg_theory_clean = compute_theoretical_cl_kg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source, b1)

    # Debug output
    print(f"    C_l^kk theory range: [{float(jnp.nanmin(cl_kk_theory_clean)):.2e}, {float(jnp.nanmax(cl_kk_theory_clean)):.2e}]")
    print(f"    C_l^gg theory range: [{float(jnp.nanmin(cl_gg_theory_clean)):.2e}, {float(jnp.nanmax(cl_gg_theory_clean)):.2e}]")
    print(f"    C_l^kg theory range: [{float(jnp.nanmin(cl_kg_theory_clean)):.2e}, {float(jnp.nanmax(cl_kg_theory_clean)):.2e}]")

    print("  ✓ Theoretical spectra computed")

    # =========================================================================
    # PLOT
    # =========================================================================
    print("\n" + "-" * 40)
    print("Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get N_l at the clean ell values for Theory + N_l curve
    cmb_nell_cfg = model_config.get("cmb_noise_nell", {})
    if isinstance(cmb_nell_cfg, str):
        # File path - PR4 format is 1D (one N_l value per line, ell = line index starting at 0)
        nell_data = np.loadtxt(cmb_nell_cfg)
        if nell_data.ndim == 1:
            # 1D: ell is implied (0, 1, 2, ...)
            nell_ell = np.arange(len(nell_data))
            nell_val = nell_data
        else:
            # 2D: first column is ell
            nell_ell, nell_val = nell_data[:, 0], nell_data[:, 1]
    elif isinstance(cmb_nell_cfg, dict):
        nell_ell = np.array(cmb_nell_cfg["ell"])
        nell_val = np.array(cmb_nell_cfg["N_ell"])
    else:
        nell_ell = np.arange(10, 3000)
        nell_val = np.ones_like(nell_ell) * 1e-8  # Fallback
    nell_at_ell_clean = np.interp(ell_clean, nell_ell, nell_val)

    # Filter valid ell values for plotting (remove NaN)
    valid_idx = np.isfinite(ell) & np.isfinite(cl_kk_box_mean) & (ell > 0)
    ell_valid = ell[valid_idx]

    # --- SUBPLOT 1: Kappa Auto ---
    plt.sca(axes[0])
    # Mean with error band
    if n_realizations > 1:
        plt.fill_between(ell_valid,
                         (cl_kk_box_mean - cl_kk_box_std)[valid_idx],
                         (cl_kk_box_mean + cl_kk_box_std)[valid_idx],
                         color='blue', alpha=0.2)
        plt.fill_between(ell_valid,
                         (cl_kk_obs_mean - cl_kk_obs_std)[valid_idx],
                         (cl_kk_obs_mean + cl_kk_obs_std)[valid_idx],
                         color='gray', alpha=0.15)
    plt.loglog(ell_valid, cl_kk_box_mean[valid_idx], '-', color='blue', lw=1.5,
               label=rf"$C_\ell^{{\kappa \kappa}}$ (Mean ± 1σ, N={n_realizations})")
    plt.loglog(ell_valid, cl_kk_obs_mean[valid_idx], '-', color='gray', alpha=0.7, lw=1.5,
               label=r"Signal + Planck PR4 $N_\ell$")
    plt.plot(ell_clean, np.array(cl_kk_theory_clean), '--', color='blue', alpha=0.8, lw=2, label=r"Theory")
    plt.plot(ell_clean, np.array(cl_kk_theory_clean) + nell_at_ell_clean, '--', color='gray', alpha=0.8, lw=2, label=r"Theory + $N_\ell$")

    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title(r"CMB Convergence $C_\ell^{\kappa \kappa}$")
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # --- SUBPLOT 2: Galaxy & Cross ---
    plt.sca(axes[1])
    valid_idx_gg = np.isfinite(ell) & np.isfinite(cl_gg_mean) & (ell > 0)
    ell_valid_gg = ell[valid_idx_gg]
    if n_realizations > 1:
        plt.fill_between(ell_valid_gg,
                         (cl_gg_mean - cl_gg_std)[valid_idx_gg],
                         (cl_gg_mean + cl_gg_std)[valid_idx_gg],
                         color='green', alpha=0.2)
        plt.fill_between(ell_valid_gg,
                         np.abs(cl_kg_mean - cl_kg_std)[valid_idx_gg],
                         np.abs(cl_kg_mean + cl_kg_std)[valid_idx_gg],
                         color='red', alpha=0.2)
    plt.loglog(ell_valid_gg, cl_gg_mean[valid_idx_gg], '-', color='green', lw=1.5,
               label=rf"$C_\ell^{{gg}}$ (Mean, N={n_realizations})")
    plt.loglog(ell_valid_gg, np.abs(cl_kg_mean)[valid_idx_gg], '-', color='red', lw=1.5,
               label=r"|$C_\ell^{\kappa g}$| (Mean)")
    plt.plot(ell_clean, np.array(cl_gg_theory_clean), '--', color='green', alpha=0.8, lw=2, label=r"Theory $C_\ell^{gg}$")
    plt.plot(ell_clean, np.abs(np.array(cl_kg_theory_clean)), '--', color='red', alpha=0.8, lw=2, label=r"Theory |$C_\ell^{\kappa g}$|")

    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title(r"Galaxy Auto & Cross Spectra")
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Add info text (moved lower to avoid overlapping with data)
    info_text = (
        f"Box: {model_config['box_shape']} Mpc/h | Mesh: {model_config['mesh_shape']} | "
        f"Cell: {cell_size} Mpc/h | Field: {field_size:.1f}° | N={n_realizations}"
    )
    fig.text(0.5, 0.01, info_text, fontsize=8, family='monospace', ha='center',
             bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    plt.tight_layout()

    # Save
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cl_spectra_{timestamp}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    if args.show:
        plt.show()
    else:
        plt.close()

    # =========================================================================
    # 2D Maps
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes2[0].imshow(np.array(kappa_box), origin='lower', cmap='RdBu_r')
    axes2[0].set_title(r"$\kappa$ (box signal)")
    plt.colorbar(im0, ax=axes2[0])

    im1 = axes2[1].imshow(np.array(kappa_obs), origin='lower', cmap='RdBu_r')
    axes2[1].set_title(r"$\kappa$ (observed = signal + noise)")
    plt.colorbar(im1, ax=axes2[1])

    im2 = axes2[2].imshow(np.array(gxy_delta), origin='lower', cmap='viridis')
    axes2[2].set_title(r"Galaxy $\delta_g$ (projected)")
    plt.colorbar(im2, ax=axes2[2])

    plt.tight_layout()

    output_file2 = output_dir / f"maps_2d_{timestamp}.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file2}")

    if args.show:
        plt.show()
    else:
        plt.close()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
