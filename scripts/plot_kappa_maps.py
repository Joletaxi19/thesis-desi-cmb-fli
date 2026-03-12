#!/usr/bin/env python3
"""
Plot CMB convergence κ maps from a single forward-model realization.

In closure mode: generates truth via model.predict() and shows:
    - κ observed (= κ_pred + noise)
    - κ predicted (noiseless, from Born projection)
    - Galaxy density projected (if galaxies_enabled)

In abacus mode: loads the AbacusSummit κ map and shows:
    - κ observed (Abacus + N_ℓ noise)
    - κ Abacus (noiseless)

Reads all parameters from configs/inference/config.yaml (same as run_inference.py).

Usage:
    python scripts/plot_kappa_maps.py
    python scripts/plot_kappa_maps.py --config configs/inference/config.yaml --seed 42
    python scripts/plot_kappa_maps.py --show
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot κ maps from a single forward-model realization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/inference/config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip HEALPix alm cut in abacus mode (diagnostic)")
    return parser.parse_args()


def main():
    args = parse_args()

    from desi_cmb_fli import utils
    from desi_cmb_fli.model import get_model_from_config

    cfg = utils.yload(args.config)
    observation_mode = cfg.get("observation_mode", "closure")
    truth_params = cfg.get("truth_params", {})
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    smooth = not args.no_smooth

    print(f"Config: {args.config}")
    print(f"Mode: {observation_mode}, seed: {seed}")

    # ── Build model ──────────────────────────────────────────────────────────
    model, model_config = get_model_from_config(cfg)
    cmb_enabled = model.cmb_enabled
    galaxies_enabled = model.galaxies_enabled

    if not cmb_enabled:
        print("CMB lensing is disabled — nothing to plot.")
        return

    field_size = model.cmb_field_size_deg
    npix = model.cmb_field_npix
    chi_center = float(model.box_center[2]) if hasattr(model, "box_center") else float(model.box_shape[2] / 2.0)

    # ── Generate / load κ maps ───────────────────────────────────────────────
    if observation_mode == "abacus":
        print("Loading AbacusSummit κ map...")
        import asdf as _asdf

        from desi_cmb_fli.cmb_lensing import prepare_abacus_kappa

        abacus_cfg = cfg.get("abacus_kappa", {})
        abacus_file = abacus_cfg.get(
            "file",
            "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/"
            "v1/AbacusSummit_base_c000_ph000/kappa_00047.asdf",
        )
        lon = float(abacus_cfg.get("lon", 75.0))
        lat = float(abacus_cfg.get("lat", 20.0))

        with _asdf.open(abacus_file) as _f:
            kappa_hp = np.array(_f.tree["data"]["kappa"])

        if smooth:
            kappa_noiseless = prepare_abacus_kappa(kappa_hp, lon, lat, npix, field_size)
        else:
            import healpy as hp
            reso = field_size * 60.0 / npix
            kappa_noiseless = np.asarray(hp.gnomview(
                kappa_hp, rot=[lon, lat], reso=reso, xsize=npix,
                return_projected_map=True, no_plot=True,
            ))
        del kappa_hp

        # Add noise realization
        field_area_sr = (field_size * jnp.pi / 180.0) ** 2
        var_k = model.nell_grid * (npix**4 / field_area_sr)
        var_k = jnp.where(jnp.isinf(var_k) | jnp.isnan(var_k), 0.0, var_k)
        eps_k = jnp.fft.fftn(jr.normal(jr.key(seed), shape=(npix, npix)))
        noise_k = eps_k * jnp.sqrt(var_k) / npix
        noise_k = noise_k.at[0, 0].set(0.0)
        noise_map = jnp.fft.ifftn(noise_k).real

        kappa_pred = jnp.array(kappa_noiseless)
        kappa_obs = kappa_pred + noise_map
        obs_field = None

        print(f"  Patch: lon={lon:.0f}°, lat={lat:.0f}°, {field_size:.1f}°, {npix}px")
        print(f"  κ std(noiseless): {float(jnp.std(kappa_pred)):.4f}")
        print(f"  κ std(observed):  {float(jnp.std(kappa_obs)):.4f}")

    else:
        print("Running forward model (closure)...")
        truth = model.predict(
            samples=truth_params,
            hide_base=False, hide_samp=False, hide_det=False,
            frombase=True, rng=jr.key(seed),
        )
        kappa_pred = truth.get("kappa_pred")
        kappa_obs = truth.get("kappa_obs")
        obs_field = truth.get("obs") if galaxies_enabled else None

        if kappa_pred is not None:
            print(f"  κ std(predicted): {float(jnp.std(kappa_pred)):.4f}")
        if kappa_obs is not None:
            print(f"  κ std(observed):  {float(jnp.std(kappa_obs)):.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    has_galaxy = obs_field is not None
    ncols = 3 if has_galaxy else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 2:
        axes = list(axes)

    # Panel 1: κ observed
    obs_arr = np.asarray(kappa_obs)
    vmax_obs = float(np.percentile(np.abs(obs_arr), 99))
    obs_title = ("κ observed (Abacus + $N_\\ell$ noise)" if observation_mode == "abacus"
                 else "κ observed (pred + noise)")
    im0 = axes[0].imshow(obs_arr, origin="lower", cmap="RdBu_r",
                         vmin=-vmax_obs, vmax=vmax_obs,
                         extent=[0, field_size, 0, field_size])
    axes[0].set_title(obs_title)
    axes[0].set_xlabel("θ [deg]")
    axes[0].set_ylabel("θ [deg]")
    plt.colorbar(im0, ax=axes[0], label="κ")

    # Panel 2: κ predicted (noiseless)
    pred_arr = np.asarray(kappa_pred)
    vmax_pred = float(np.percentile(np.abs(pred_arr), 99))
    pred_title = ("κ Abacus (noiseless)" if observation_mode == "abacus"
                  else "κ predicted (noiseless)")
    im1 = axes[1].imshow(pred_arr, origin="lower", cmap="RdBu_r",
                         vmin=-vmax_pred, vmax=vmax_pred,
                         extent=[0, field_size, 0, field_size])
    axes[1].set_title(pred_title)
    axes[1].set_xlabel("θ [deg]")
    plt.colorbar(im1, ax=axes[1], label="κ")

    # Panel 3: Galaxy density projected (closure + galaxies only)
    if has_galaxy:
        from desi_cmb_fli.cmb_lensing import project_flat_sky
        gxy_proj = project_flat_sky(
            np.array(obs_field), model.box_shape,
            field_size, npix, chi_center,
        )
        im2 = axes[2].imshow(gxy_proj, origin="lower", cmap="viridis",
                             extent=[0, field_size, 0, field_size])
        axes[2].set_title("Galaxy density (projected)")
        axes[2].set_xlabel("θ [deg]")
        plt.colorbar(im2, ax=axes[2], label="N [counts]")

    # Info footer
    box = model.box_shape
    cell = model.cell_shape[0] if hasattr(model, "cell_shape") else 0
    info = (f"Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}] Mpc/h | "
            f"Cell: {cell:.1f} Mpc/h | Field: {field_size:.1f}° | "
            f"{npix}×{npix} pix | seed={seed}")
    fig.suptitle(info, fontsize=9, y=0.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"kappa_maps_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {outfile}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
