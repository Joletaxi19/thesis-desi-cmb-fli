#!/usr/bin/env python
"""
Quick 3D P(k) Diagnostic Script

Measures the 3D matter power spectrum from the model's matter_mesh field
and compares it to the theoretical jax_cosmo nonlinear P(k) at the effective
redshift. This isolates the 3D painting quality from Any projection effects.

Usage:
    python scripts/quick_pk_spectra.py --n_realizations 5
    python scripts/quick_pk_spectra.py --config configs/inference/config.yaml --cell_size 10.0
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.metrics import spectrum
from desi_cmb_fli.model import get_model_from_config

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick 3D P(k) diagnostic from simulated matter field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/inference/config.yaml")
    parser.add_argument("--cell_size", type=float, default=None,
                        help="Override cell size in Mpc/h")
    parser.add_argument("--n_realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def pk_theory(cosmo, k_hmpc, a):
    """Nonlinear P(k) in (Mpc/h)^3 at scale factor a, evaluated at k in h/Mpc."""
    pk_fn = jax.vmap(lambda ki: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, a)))
    return np.array(pk_fn(jnp.array(k_hmpc, dtype=float)))


def main():
    args = parse_args()

    print("=" * 72)
    print("QUICK 3D P(k) DIAGNOSTIC")
    print("=" * 72)
    print(f"\nJAX backend: {jax.default_backend()}")

    cfg_dict = utils.yload(args.config)
    if args.cell_size is not None:
        cfg_dict["model"]["cell_size"] = args.cell_size

    model, model_config = get_model_from_config(cfg_dict)
    truth_params = cfg_dict.get("truth_params", {})
    base_seed = args.seed if args.seed is not None else cfg_dict.get("seed", 42)
    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    cosmo = get_cosmology(**truth_params)
    a_eff = float(model.a_fid)
    print(f"\nEffective scale factor: a_fid={a_eff:.4f}  (z={1/a_eff - 1:.4f})")
    print(f"Mesh shape:  {tuple(model.mesh_shape)}")
    print(f"Box shape:   {tuple(model.box_shape)} Mpc/h")
    print(f"Cell size:   {float(model.cell_shape[0]):.3f} Mpc/h")
    k_fund = float(2 * np.pi / np.min(model.box_shape))
    k_nyq  = float(np.pi * np.min(model.mesh_shape / model.box_shape))
    print(f"k_fund={k_fund:.4f}  k_Nyq={k_nyq:.4f} h/Mpc")
    print(f"paint_oversamp={model.paint_oversamp}")

    n_real = args.n_realizations
    print(f"\nGenerating {n_real} realization(s)...\n")

    all_k, all_pk = [], []

    for i in range(n_real):
        seed_i = base_seed + i
        print(f"  Realization {i+1}/{n_real} (seed={seed_i})", end="\r")
        truth_i = model.predict(
            samples=truth_params,
            hide_base=False, hide_samp=False, hide_det=False,
            frombase=True, rng=jr.key(seed_i),
        )

        mm = np.array(truth_i["matter_mesh"])
        # matter_mesh = (1+delta) after normalization: mean≈1, std≈sigma_8_eff
        delta = mm / np.mean(mm) - 1.0

        k3d, pk3d = spectrum(delta, box_shape=np.array(model.box_shape), comp=(0, 0))
        all_k.append(np.array(k3d))
        all_pk.append(np.array(pk3d))

    print(f"  Done.{' '*40}")

    k_arr  = all_k[0]
    pk_mat = np.array(all_pk)        # shape (n_real, n_bins)
    pk_mean = pk_mat.mean(axis=0)
    pk_std  = pk_mat.std(axis=0)

    pk_th = pk_theory(cosmo, k_arr, a_eff)

    ratio = pk_mean / pk_th

    # ── Text diagnostics ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("P(k) RATIO TABLE:  measured (mean±std) / theory")
    print("=" * 72)
    low_k  = k_arr < 0.5 * k_nyq
    high_k = (k_arr >= 0.5 * k_nyq) & (k_arr < k_nyq)
    print(f"  {'k [h/Mpc]':>12s}  {'P(k)_meas':>12s}  {'P(k)_th':>12s}  {'ratio':>8s}  {'std/mean':>8s}")
    for i in range(len(k_arr)):
        flag = "<-- Nyq" if not low_k[i] else ""
        print(f"  {k_arr[i]:12.4f}  {pk_mean[i]:12.4e}  {pk_th[i]:12.4e}  {ratio[i]:8.4f}  "
              f"{pk_std[i]/pk_mean[i]:8.3f}  {flag}")
    print(f"\n  Mean ratio  k < 0.5 k_Nyq : {np.nanmean(ratio[low_k]):.4f}")
    print(f"  Mean ratio  k > 0.5 k_Nyq : {np.nanmean(ratio[high_k]):.4f}")
    print("  (Should be ~1 everywhere if painting is correct)")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax_pk, ax_rat) = plt.subplots(2, 1, figsize=(8, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

    # Individual realizations (faded)
    for pk_i in all_pk:
        ax_pk.loglog(k_arr, pk_i, color="steelblue", lw=0.5, alpha=0.4)

    # Mean ± std band
    ax_pk.fill_between(k_arr, pk_mean - pk_std, pk_mean + pk_std,
                       color="steelblue", alpha=0.3, label=r"Mean ± 1$\sigma$")
    ax_pk.loglog(k_arr, pk_mean, color="steelblue", lw=2, label="Measured mean")

    # Theory
    k_fine = np.geomspace(k_arr[0] * 0.9, k_arr[-1] * 1.1, 200)
    pk_th_fine = pk_theory(cosmo, k_fine, a_eff)
    ax_pk.loglog(k_fine, pk_th_fine, "k--", lw=1.5, label=f"Theory (jax_cosmo, a={a_eff:.3f})")

    # k_Nyq marker
    ax_pk.axvline(k_nyq, color="red", lw=1, ls=":", alpha=0.7, label=f"$k_{{Nyq}}={k_nyq:.3f}$")
    ax_pk.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.7,
                  label=f"$0.5 k_{{Nyq}}={0.5*k_nyq:.3f}$")

    ax_pk.set_ylabel(r"$P(k)\;[({\rm Mpc}/h)^3]$", fontsize=12)
    title = (f"3D Matter P(k) — {n_real} realization(s)\n"
             f"Mesh {tuple(model.mesh_shape)}, Box {tuple(int(b) for b in model.box_shape)} Mpc/h, "
             f"paint_oversamp={model.paint_oversamp}")
    ax_pk.set_title(title, fontsize=10)
    ax_pk.legend(fontsize=9, loc="lower left")
    ax_pk.grid(True, alpha=0.2)
    ax_pk.set_ylim(bottom=1e0)

    # Ratio panel
    ax_rat.semilogx(k_arr, ratio, "o-", color="steelblue", ms=4)
    ax_rat.axhline(1.0, color="k", lw=1.2)
    ax_rat.axhline(1.05, color="k", lw=0.5, ls="--")
    ax_rat.axhline(0.95, color="k", lw=0.5, ls="--")
    ax_rat.axvline(k_nyq, color="red", lw=1, ls=":", alpha=0.7)
    ax_rat.axvline(0.5 * k_nyq, color="orange", lw=1, ls=":", alpha=0.7)
    ax_rat.fill_between(k_arr, ratio - pk_std / pk_th, ratio + pk_std / pk_th,
                        color="steelblue", alpha=0.25)
    ax_rat.set_xlabel(r"$k\;[h/{\rm Mpc}]$", fontsize=12)
    ax_rat.set_ylabel("Meas / Theory", fontsize=10)
    ax_rat.set_ylim(0.5, 1.5)
    ax_rat.grid(True, alpha=0.2)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"pk_spectra_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {outfile}")
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
