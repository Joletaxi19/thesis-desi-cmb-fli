#!/usr/bin/env python
"""
Quick C_l Spectra Diagnostic Script

Generate C_l power spectra (kappa-kappa, galaxy-kappa, galaxy-galaxy) from
a simulation using the same config.yaml as run_inference.py.

This allows fast visual verification of spectra without running full inference.
Generates N realizations via model.predict(), accumulates the measured spectra,
and produces one figure with mean ± std bands.

Usage:
    python scripts/quick_cl_spectra.py --config configs/inference/config.yaml --cell_size 5.0 --n_realizations 20
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    NYQUIST_FRACTION,
    compute_cl_high_z,
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kg,
    prepare_abacus_kappa,
)
from desi_cmb_fli.metrics import spectrum as spectrum_3d
from desi_cmb_fli.model import get_model_from_config
from desi_cmb_fli.validation import measure_spectra

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
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Skip HEALPix alm cut in abacus mode (diagnostic only — exposes gnomview aliasing)"
    )

    return parser.parse_args()


def _build_abacus_proxy(cfg_dict, cell_size):
    """Lightweight Abacus proxy for abacus + cell_size mode: geometry + N_ell only, without FieldLevelModel."""
    import types
    model_cfg = cfg_dict.get("model", {})
    cmb_cfg   = cfg_dict.get("cmb_lensing", {})
    box       = np.array(model_cfg["box_shape"], dtype=float)   # [Lx, Ly, Lz] Mpc/h
    npix      = int(round(box[0] / cell_size))
    chi_max   = float(box[2])
    fsize_deg = float(np.degrees(2 * np.arctan(box[0] / (2 * chi_max))))
    freq      = np.fft.fftfreq(npix, d=fsize_deg / npix) * 360.0
    lx, ly    = np.meshgrid(freq, freq, indexing="ij")
    ell_grid  = np.sqrt(lx**2 + ly**2)

    data      = np.loadtxt(cmb_cfg["cmb_noise_nell"])
    m         = (data[:, 0] > 0) & np.isfinite(data[:, 1]) & (data[:, 1] > 0)
    nell_grid = np.exp(np.interp(np.log(np.maximum(ell_grid, 1e-5)),
                                 np.log(data[m, 0]), np.log(data[m, 1])))
    nell_grid *= float(cmb_cfg.get("cmb_noise_scaling", 1.0))
    nell_grid[0, 0] = np.inf

    print(f"[Abacus proxy] cell_size={cell_size} Mpc/h → npix={npix}, "
          f"field_size={fsize_deg:.2f}°, ℓ_Nyq={180*npix/fsize_deg:.0f}")

    proxy = types.SimpleNamespace(
        cmb_enabled=True, galaxies_enabled=False,
        cmb_field_size_deg=fsize_deg, cmb_field_npix=npix,
        cmb_z_source=float(cmb_cfg.get("z_source", 1100.0)),
        ell_grid=ell_grid, ell_nyquist=180.0 * npix / fsize_deg,
        nell_grid=nell_grid, box_shape=box,
        box_center=np.array([0., 0., chi_max / 2]),
        cell_shape=np.array([cell_size, cell_size, cell_size]),
        full_los_correction=bool(cmb_cfg.get("full_los_correction", False)),
        chi_high_z_max=cmb_cfg.get("chi_high_z_max", None),
        high_z_mode=cmb_cfg.get("high_z_mode", "exact"),
        cl_high_z_cached=None, high_z_gradients=None,
        loc_fid={k: float(v["loc_fid"]) for k, v in cfg_dict.get("latents", {}).items()
                 if isinstance(v, dict) and "loc_fid" in v},
    )
    return proxy, dict(model_cfg, cell_size=cell_size)


def main():
    args = parse_args()

    print("=" * 80)
    print("QUICK C_l SPECTRA DIAGNOSTIC")
    print("=" * 80)
    print(f"\nJAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    smooth = not args.no_smooth

    # Load config dict
    cfg_dict = utils.yload(args.config)
    observation_mode = cfg_dict.get("observation_mode", "closure")

    if observation_mode == "abacus":
        cell_size_proxy = args.cell_size or float(cfg_dict["model"]["cell_size"])
        model, model_config = _build_abacus_proxy(cfg_dict, cell_size_proxy)
    else:
        if args.cell_size is not None:
            cfg_dict["model"]["cell_size"] = args.cell_size
        model, model_config = get_model_from_config(cfg_dict)

    truth_params = cfg_dict.get("truth_params", {})
    base_seed = args.seed if args.seed is not None else cfg_dict.get("seed", 42)
    output_dir = Path(args.output_dir or "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_real = args.n_realizations

    # ── Build truth dicts (mode-dependent) ──────────────────────────────────
    all_cl_kk_pred, all_cl_kk_obs, all_cl_gg, all_cl_kg = [], [], [], []
    # Binned accumulators
    all_cl_kk_pred_b, all_cl_kk_obs_b, all_cl_gg_b, all_cl_kg_b = [], [], [], []
    # 3D P(k) accumulators for painting diagnostic
    all_pk_matter = []
    ell = None
    ell_b = None

    if observation_mode == "abacus":
        # Load Abacus kappa
        print("\n[Abacus mode] Loading Abacus kappa map...")
        import asdf as _asdf
        import healpy as hp

        abacus_cfg = cfg_dict.get("abacus_kappa", {})
        abacus_file = abacus_cfg.get(
            "file",
            "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/"
            "v1/AbacusSummit_base_c000_ph000/kappa_00047.asdf",
        )
        lon_patch = float(abacus_cfg.get("lon", 75.0))
        lat_patch = float(abacus_cfg.get("lat", 20.0))

        with _asdf.open(abacus_file) as _f:
            _kappa_hp = np.array(_f.tree["data"]["kappa"])
        if smooth:
            _patch_np = prepare_abacus_kappa(
                _kappa_hp, lon_patch, lat_patch,
                model.cmb_field_npix, model.cmb_field_size_deg,
            )
        else:
            print("[Abacus] Skipping band-limit (--no-smooth) — diagnostic mode.")
            _reso = model.cmb_field_size_deg * 60.0 / model.cmb_field_npix
            _patch_np = np.asarray(hp.gnomview(
                _kappa_hp, rot=[lon_patch, lat_patch],
                reso=_reso, xsize=model.cmb_field_npix,
                return_projected_map=True, no_plot=True,
            ))
        del _kappa_hp

        # AbacusSummit uses 0 (not hp.UNSEEN) as sentinel for uncovered pixels.
        n_zero   = int(np.sum(_patch_np == 0.0))
        zero_frac = n_zero / _patch_np.size
        if zero_frac > 0.001:
            raise ValueError(
                f"[Abacus] {n_zero} pixels ({zero_frac:.2%}) are exactly zero — "
                "likely outside the AbacusSummit lightcone footprint. "
                "Reduce field size (box_shape Lx/Ly) or adjust abacus_kappa.lon/lat."
            )

        # Print patch diagnostics
        print(f"\n{'='*60}")
        print("  ABACUS PATCH DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"  shape   = {_patch_np.shape}")
        print(f"  zeros   = {n_zero} ({zero_frac:.4%})")
        print(f"  min(κ)  = {float(_patch_np.min()):.6f}")
        print(f"  max(κ)  = {float(_patch_np.max()):.6f}")
        print(f"  mean(κ) = {float(_patch_np.mean()):.6f}")
        print(f"  std(κ)  = {float(_patch_np.std()):.6f}")
        print(f"  p1/p99  = {float(np.percentile(_patch_np, 1)):.6f} / "
              f"{float(np.percentile(_patch_np, 99)):.6f}")
        print(f"{'='*60}")

        kappa_noiseless = jnp.array(_patch_np)
        del _patch_np
        print(f"[Abacus mode] Loaded patch: shape={kappa_noiseless.shape}, "
              f"std={float(jnp.std(kappa_noiseless)):.4f}")

        _npix = model.cmb_field_npix
        _field_area_sr = (model.cmb_field_size_deg * jnp.pi / 180.0) ** 2
        _var_k = model.nell_grid * (_npix**4 / _field_area_sr)
        _var_k = jnp.where(jnp.isinf(_var_k) | jnp.isnan(_var_k), 0.0, _var_k)

        print(f"Generating {n_real} noise realization(s) on Abacus kappa...")
        for i in range(n_real):
            seed_i = base_seed + i
            print(f"  Realization {i+1}/{n_real} (noise seed={seed_i})", end="\r")
            _eps_k   = jnp.fft.fftn(jr.normal(jr.key(seed_i), shape=(_npix, _npix)))
            _noise_k = _eps_k * jnp.sqrt(_var_k) / _npix
            _noise_k = _noise_k.at[0, 0].set(0.0)
            _noise_map = jnp.fft.ifftn(_noise_k).real
            truth_i = {
                "kappa_pred": kappa_noiseless,
                "kappa_obs": kappa_noiseless + _noise_map,
            }
            sp = measure_spectra(truth_i, model, model_config)
            if ell is None:
                ell = sp["ell"]
            if ell_b is None and sp["ell_b"] is not None:
                ell_b = sp["ell_b"]
            for key, lst in [("cl_kk_pred", all_cl_kk_pred),
                              ("cl_kk_obs", all_cl_kk_obs),
                              ("cl_gg", all_cl_gg),
                              ("cl_kg", all_cl_kg)]:
                if sp[key] is not None:
                    lst.append(sp[key])
            for key, lst in [("cl_kk_pred_b", all_cl_kk_pred_b),
                              ("cl_kk_obs_b", all_cl_kk_obs_b),
                              ("cl_gg_b", all_cl_gg_b),
                              ("cl_kg_b", all_cl_kg_b)]:
                if sp[key] is not None:
                    lst.append(sp[key])

    else:
        # Closure: N independent LPT realizations
        print(f"\n[Closure mode] Generating {n_real} realization(s)...")
        for i in range(n_real):
            seed_i = base_seed + i
            print(f"  Realization {i+1}/{n_real} (seed={seed_i})", end="\r")
            truth_i = model.predict(
                samples=truth_params,
                hide_base=False,
                hide_samp=False,
                hide_det=False,
                frombase=True,
                rng=jr.key(seed_i),
            )
            sp = measure_spectra(truth_i, model, model_config)
            # 3D P(k) diagnostic: measure matter overdensity spectrum directly
            if "matter_mesh" in truth_i:
                mm = np.array(truth_i["matter_mesh"])
                delta_3d = mm / np.mean(mm) - 1.0
                k3d, pk3d = spectrum_3d(delta_3d, box_shape=np.array(model.box_shape), comp=(0, 0))
                all_pk_matter.append((np.array(k3d), np.array(pk3d)))
            if ell is None:
                ell = sp["ell"]
            if ell_b is None and sp["ell_b"] is not None:
                ell_b = sp["ell_b"]
            for key, lst in [("cl_kk_pred", all_cl_kk_pred),
                              ("cl_kk_obs", all_cl_kk_obs),
                              ("cl_gg", all_cl_gg),
                              ("cl_kg", all_cl_kg)]:
                if sp[key] is not None:
                    lst.append(sp[key])
            for key, lst in [("cl_kk_pred_b", all_cl_kk_pred_b),
                              ("cl_kk_obs_b", all_cl_kk_obs_b),
                              ("cl_gg_b", all_cl_gg_b),
                              ("cl_kg_b", all_cl_kg_b)]:
                if sp[key] is not None:
                    lst.append(sp[key])
    print("")

    if ell is None:
        print("No spectra measured — nothing to plot.")
        return

    # ── Compute mean and std ─────────────────────────────────────────────────
    def _stats(lst):
        if not lst:
            return None, None
        arr = np.array(lst)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    cl_kk_pred_mean, cl_kk_pred_std = _stats(all_cl_kk_pred)
    cl_kk_obs_mean, cl_kk_obs_std = _stats(all_cl_kk_obs)
    cl_gg_mean, cl_gg_std = _stats(all_cl_gg)
    cl_kg_mean, cl_kg_std = _stats(all_cl_kg)

    # Binned stats
    cl_kk_pred_b_mean, cl_kk_pred_b_std = _stats(all_cl_kk_pred_b)
    cl_kk_obs_b_mean, cl_kk_obs_b_std = _stats(all_cl_kk_obs_b)
    cl_gg_b_mean, cl_gg_b_std = _stats(all_cl_gg_b)
    cl_kg_b_mean, cl_kg_b_std = _stats(all_cl_kg_b)

    # ── Theoretical Limber spectra ───────────────────────────────────────────
    print("Computing theoretical spectra...")
    # In Abacus mode, use the Abacus truth cosmology for theory curves
    if observation_mode == "abacus":
        cosmo_for_theory = cfg_dict.get("abacus_truth_params", truth_params)
        if cosmo_for_theory is truth_params:
            print("  WARNING: abacus_truth_params not found in config, falling back to truth_params")
    else:
        cosmo_for_theory = truth_params
    cosmo_val = get_cosmology(**cosmo_for_theory)
    cmb_enabled = model.cmb_enabled
    z_source = model.cmb_z_source
    chi_min, chi_max = 1.0, float(model.box_shape[2])
    b1_lag = cosmo_for_theory.get("b1", truth_params.get("b1", 1.0))
    bE = 1.0 + b1_lag
    # 3D Nyquist wavenumber: used to make Limber theory resolution-aware
    dx_mesh = float(model.box_shape[0]) / float(model.mesh_shape[0])
    k_nyq_mesh = np.pi / dx_mesh

    if cmb_enabled:
        field_size = model.cmb_field_size_deg
        npix = model.cmb_field_npix
    else:
        half_angle_rad = np.arctan(model.box_shape[0] / (2.0 * model.box_shape[2]))
        field_size = float(2.0 * half_angle_rad * (180.0 / np.pi))
        npix = 256
    ell_nyquist = np.pi * npix / (field_size * np.pi / 180.0)

    ell_theory = np.geomspace(10, NYQUIST_FRACTION * ell_nyquist, 100)

    cl_kk_theory, cl_gg_theory, cl_kg_theory = None, None, None
    cl_gg_theory_full, cl_kg_theory_full = None, None  # uncorrected reference
    nell_at_theory = np.zeros_like(ell_theory)
    cl_high_z_theory = np.zeros_like(ell_theory)

    if cmb_enabled:
        cl_kk_theory = np.asarray(compute_theoretical_cl_kappa(
            cosmo_val, jnp.array(ell_theory), chi_min, chi_max, z_source
        ))

        ell_grid_flat = model.ell_grid.flatten()
        sort_idx = np.argsort(ell_grid_flat)
        nell_at_theory = np.exp(np.interp(
            np.log(np.maximum(ell_theory, 1e-5)),
            np.log(np.maximum(ell_grid_flat[sort_idx], 1e-5)),
            np.log(model.nell_grid.flatten()[sort_idx])
        ))

        if model.full_los_correction:
            _high_z_mode = 'exact' if observation_mode == 'abacus' else model.high_z_mode
            cl_high_z_grid = compute_cl_high_z(
                cosmo_val, model.ell_grid, model.box_shape[2], model.chi_high_z_max, z_source,
                mode=_high_z_mode, cl_cached=model.cl_high_z_cached,
                gradients=model.high_z_gradients, loc_fid=model.loc_fid,
            )
            cl_high_z_flat = np.array(cl_high_z_grid).flatten()
            cl_high_z_theory = np.interp(ell_theory, ell_grid_flat[sort_idx], cl_high_z_flat[sort_idx])



    if cl_gg_mean is not None:
        # Resolution-aware: exclude shells below the 3D Nyquist limit per ell
        cl_gg_theory = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, jnp.array(ell_theory), chi_min, chi_max, bE, k_nyq=k_nyq_mesh
        ))
        # Full-resolution reference (no k_nyq cut) for comparison
        cl_gg_theory_full = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, jnp.array(ell_theory), chi_min, chi_max, bE
        ))
        # Poisson shot-noise level: N_ell = Omega_sky / N_gal
        # N_gal = n_bar * V_box  (number of galaxies in the simulated volume)
        # Omega_sky = (field_size_deg * pi/180)^2  (angular area in sr)
        _field_size_deg = field_size if cmb_enabled else (
            2.0 * np.degrees(np.arctan(float(model.box_shape[0]) / (2.0 * float(model.box_shape[2]))))
        )
        _field_area_sr = (_field_size_deg * np.pi / 180.0) ** 2
        _n_gal = float(model.gxy_density) * float(np.prod(model.box_shape))
        cl_gg_shot = _field_area_sr / _n_gal
        print(f"  Galaxy shot noise: N_ell = {cl_gg_shot:.3e} sr"
              f"  (n_bar={model.gxy_density:.2e} (Mpc/h)^-3, "
              f"N_gal~{_n_gal:.0f}, Omega={_field_area_sr*1e4:.2f}e-4 sr)")
        if cmb_enabled:
            cl_kg_theory = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, jnp.array(ell_theory), chi_min, chi_max, z_source, bE, k_nyq=k_nyq_mesh
            ))
            cl_kg_theory_full = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, jnp.array(ell_theory), chi_min, chi_max, z_source, bE
            ))

    # ── Quantitative diagnostics: measured/theory ratio at binned ell ───────
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Binned measured / theory ratio")
    print("=" * 80)

    if ell_b is not None and cl_kk_theory is not None and cl_kk_pred_b_mean is not None:
        # Interpolate box-only theory to binned ell
        cl_kk_th_at_b = np.interp(ell_b, ell_theory, cl_kk_theory)
        ratio_kk = cl_kk_pred_b_mean / cl_kk_th_at_b
        vb = np.isfinite(ratio_kk) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
        print("\n  C_l^kk (pred box, binned) / Limber(box):")
        print(f"  {'ell_b':>8s}  {'measured':>12s}  {'theory':>12s}  {'ratio':>8s}")
        for i in np.where(vb)[0]:
            print(f"  {ell_b[i]:8.1f}  {cl_kk_pred_b_mean[i]:12.4e}  {cl_kk_th_at_b[i]:12.4e}  {ratio_kk[i]:8.4f}")
        print(f"  Mean ratio (ell<200): {np.nanmean(ratio_kk[vb & (ell_b < 200)]):.4f}")
        print(f"  Mean ratio (ell>200): {np.nanmean(ratio_kk[vb & (ell_b > 200)]):.4f}")

    if ell_b is not None and cl_gg_theory is not None and cl_gg_b_mean is not None:
        cl_gg_th_at_b = np.interp(ell_b, ell_theory, cl_gg_theory)
        cl_gg_th_shot_at_b = cl_gg_th_at_b + cl_gg_shot
        ratio_gg = cl_gg_b_mean / cl_gg_th_at_b
        ratio_gg_shot = cl_gg_b_mean / cl_gg_th_shot_at_b
        vb = np.isfinite(ratio_gg) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
        print("\n  C_l^gg (binned) / Limber(gg) / Limber+shot:")
        print(f"  {'ell_b':>8s}  {'measured':>12s}  {'theory':>12s}  {'ratio':>8s}  {'ratio+shot':>10s}")
        for i in np.where(vb)[0]:
            print(f"  {ell_b[i]:8.1f}  {cl_gg_b_mean[i]:12.4e}  {cl_gg_th_at_b[i]:12.4e}"
                  f"  {ratio_gg[i]:8.4f}  {ratio_gg_shot[i]:10.4f}")
        print(f"  Mean ratio (ell<200): {np.nanmean(ratio_gg[vb & (ell_b < 200)]):.4f}"
              f"  (with shot: {np.nanmean(ratio_gg_shot[vb & (ell_b < 200)]):.4f})")
        print(f"  Mean ratio (ell>200): {np.nanmean(ratio_gg[vb & (ell_b > 200)]):.4f}"
              f"  (with shot: {np.nanmean(ratio_gg_shot[vb & (ell_b > 200)]):.4f})")

    if ell_b is not None and cl_kg_theory is not None and cl_kg_b_mean is not None:
        cl_kg_th_at_b = np.interp(ell_b, ell_theory, np.abs(np.array(cl_kg_theory)))
        ratio_kg = np.abs(cl_kg_b_mean) / cl_kg_th_at_b
        vb = np.isfinite(ratio_kg) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
        print("\n  |C_l^kg| (binned) / Limber(|kg|):")
        print(f"  {'ell_b':>8s}  {'measured':>12s}  {'theory':>12s}  {'ratio':>8s}")
        for i in np.where(vb)[0]:
            print(f"  {ell_b[i]:8.1f}  {np.abs(cl_kg_b_mean[i]):12.4e}  {cl_kg_th_at_b[i]:12.4e}  {ratio_kg[i]:8.4f}")
        print(f"  Mean ratio (ell<200): {np.nanmean(ratio_kg[vb & (ell_b < 200)]):.4f}")
        print(f"  Mean ratio (ell>200): {np.nanmean(ratio_kg[vb & (ell_b > 200)]):.4f}")

    print("=" * 80 + "\n")

    # ── Plot: one figure with binned markers + raw faded lines ─────────────
    print("Generating plot...")
    has_cmb_panel = cmb_enabled and cl_kk_pred_mean is not None
    has_gxy_panel = cl_gg_mean is not None

    if has_cmb_panel and has_gxy_panel:
        fig, (ax_kk, ax_gg) = plt.subplots(1, 2, figsize=(14, 6))
    elif has_cmb_panel:
        fig, ax_kk = plt.subplots(1, 1, figsize=(8, 6))
        ax_gg = None
    elif has_gxy_panel:
        fig, ax_gg = plt.subplots(1, 1, figsize=(8, 6))
        ax_kk = None
    else:
        print("Nothing to plot.")
        return

    valid = (ell > 10) & (ell < NYQUIST_FRACTION * ell_nyquist)
    valid_b = (ell_b is not None) and len(ell_b) > 0

    # Precompute theory combinations and high-z interpolated on measured ell grid
    cl_box_highz_theory = cl_kk_theory + cl_high_z_theory          # Box + high-z
    cl_total_theory = cl_box_highz_theory + nell_at_theory          # Box + high-z + N_ℓ

    # ── CMB κ panel ──────────────────────────────────────────────────────────
    if has_cmb_panel:
        plt.sca(ax_kk)

        # Raw unbinned as thin faded lines
        v_grey = valid & np.isfinite(cl_kk_obs_mean)
        ax_kk.loglog(ell[v_grey], cl_kk_obs_mean[v_grey], "-", color="grey", lw=0.4, alpha=0.3)
        v_pred_raw = valid & np.isfinite(cl_kk_pred_mean)
        _pred_raw_color = "steelblue" if observation_mode == "closure" else "purple"
        _pred_raw_label = r"$C_\ell^{\kappa\kappa}$ (pred)" if observation_mode == "closure" else r"$C_\ell^{\kappa\kappa}$ (Abacus, noiseless)"
        ax_kk.loglog(ell[v_pred_raw], cl_kk_pred_mean[v_pred_raw], "-", color=_pred_raw_color, lw=0.4, alpha=0.3)

        # ── Binned markers ───────────────────────────────────────────────────
        if valid_b and cl_kk_obs_b_mean is not None:
            vb = np.isfinite(cl_kk_obs_b_mean) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
            _yerr = cl_kk_obs_b_std[vb] if (cl_kk_obs_b_std is not None and n_real > 1) else None
            ax_kk.errorbar(ell_b[vb], cl_kk_obs_b_mean[vb], yerr=_yerr,
                           fmt="o", color="grey", ms=4, lw=1.2, capsize=2,
                           label=r"$C_\ell^{\kappa\kappa}$ (obs, binned)")

        if valid_b and cl_kk_pred_b_mean is not None:
            vb = np.isfinite(cl_kk_pred_b_mean) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
            if observation_mode == "closure":
                # Add high-z to binned pred
                cl_hz_on_b = np.interp(ell_b, ell_theory, cl_high_z_theory)
                _pred_b_plot = cl_kk_pred_b_mean + cl_hz_on_b
                _yerr = cl_kk_pred_b_std[vb] if (cl_kk_pred_b_std is not None and n_real > 1) else None
                ax_kk.errorbar(ell_b[vb], _pred_b_plot[vb], yerr=_yerr,
                               fmt="s", color="purple", ms=4, lw=1.2, capsize=2,
                               label=r"$C_\ell^{\kappa\kappa}$ (pred + high-$z$, binned)")
            else:
                _yerr = cl_kk_pred_b_std[vb] if (cl_kk_pred_b_std is not None and n_real > 1) else None
                ax_kk.errorbar(ell_b[vb], cl_kk_pred_b_mean[vb], yerr=_yerr,
                               fmt="s", color="purple", ms=4, lw=1.2, capsize=2,
                               label=r"$C_\ell^{\kappa\kappa}$ (Abacus, noiseless, binned)")

        # Theory dashed lines
        ax_kk.plot(ell_theory, cl_total_theory, "--", color="grey", lw=1.5, alpha=0.9,
                   label=r"Limber (Box + high-$z$ + $N_\ell$)")
        ax_kk.plot(ell_theory, cl_box_highz_theory, "--", color="purple", lw=1.5, alpha=0.9,
                   label=r"Limber (Box + high-$z$)")

        if observation_mode == "closure" and cl_kk_theory is not None:
            ax_kk.plot(ell_theory, cl_kk_theory, "--", color="steelblue", lw=1.5, alpha=0.9,
                       label=r"Limber (Box)")
            # Binned box-only pred
            if valid_b and cl_kk_pred_b_mean is not None:
                vb2 = np.isfinite(cl_kk_pred_b_mean) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
                _yerr2 = cl_kk_pred_b_std[vb2] if (cl_kk_pred_b_std is not None and n_real > 1) else None
                ax_kk.errorbar(ell_b[vb2], cl_kk_pred_b_mean[vb2], yerr=_yerr2,
                               fmt="^", color="steelblue", ms=4, lw=1, capsize=2,
                               label=r"$C_\ell^{\kappa\kappa}$ (pred box, binned)")

        ax_kk.set_xlabel(r"$\ell$")
        ax_kk.set_ylabel(r"$C_\ell$")
        mode_label = "closure" if observation_mode == "closure" else "Abacus"
        smooth_label = "" if observation_mode == "closure" else ("" if smooth else " [no smooth]")
        ax_kk.set_title(rf"$C_\ell^{{\kappa\kappa}}$ — {mode_label}{smooth_label} ({n_real} real.)")
        ax_kk.legend(fontsize=7, loc="lower left")
        ax_kk.grid(True, alpha=0.2)
        ax_kk.set_xlim(10, NYQUIST_FRACTION * ell_nyquist)

    # ── Galaxy panel ─────────────────────────────────────────────────────────
    if has_gxy_panel:
        plt.sca(ax_gg)
        v_gg = valid & np.isfinite(cl_gg_mean)

        ax_gg.loglog(ell[v_gg], cl_gg_mean[v_gg], "-", color="green", lw=0.4, alpha=0.3)
        if valid_b and cl_gg_b_mean is not None:
            vb_gg = np.isfinite(cl_gg_b_mean) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
            _yerr = cl_gg_b_std[vb_gg] if (cl_gg_b_std is not None and n_real > 1) else None
            ax_gg.errorbar(ell_b[vb_gg], cl_gg_b_mean[vb_gg], yerr=_yerr,
                           fmt="o", color="green", ms=4, lw=1.2, capsize=2,
                           label=r"$C_\ell^{gg}$ (binned)")
        if cl_gg_theory is not None:
            ax_gg.plot(ell_theory, np.array(cl_gg_theory), "--", color="green",
                       alpha=0.8, lw=2, label=r"Limber $C_\ell^{gg}$ (res.-aware)")
            ax_gg.plot(ell_theory, np.array(cl_gg_theory) + cl_gg_shot,
                       "-.", color="darkgreen", alpha=0.8, lw=1.5,
                       label=r"Limber $C_\ell^{gg}$ + shot noise")
            ax_gg.axhline(cl_gg_shot, color="gray", lw=0.8, ls=":", alpha=0.6,
                          label=rf"Shot noise $N_\ell={cl_gg_shot:.1e}$")
        if cl_gg_theory_full is not None:
            ax_gg.plot(ell_theory, np.array(cl_gg_theory_full), ":", color="green",
                       alpha=0.4, lw=1.5, label=r"Limber $C_\ell^{gg}$ (full res.)")

        if cl_kg_mean is not None:
            v_kg = valid & np.isfinite(cl_kg_mean)
            ax_gg.loglog(ell[v_kg], np.abs(cl_kg_mean)[v_kg], "-", color="red", lw=0.4, alpha=0.3)
            if valid_b and cl_kg_b_mean is not None:
                vb_kg = np.isfinite(cl_kg_b_mean) & (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
                _yerr_kg = np.abs(cl_kg_b_std[vb_kg]) if (cl_kg_b_std is not None and n_real > 1) else None
                ax_gg.errorbar(ell_b[vb_kg], np.abs(cl_kg_b_mean[vb_kg]), yerr=_yerr_kg,
                               fmt="s", color="red", ms=4, lw=1.2, capsize=2,
                               label=r"|$C_\ell^{\kappa g}$| (binned)")
            if cl_kg_theory is not None:
                ax_gg.plot(ell_theory, np.abs(np.array(cl_kg_theory)), "--", color="red",
                           alpha=0.8, lw=2, label=r"Limber |$C_\ell^{\kappa g}$| (res.-aware)")
            if cl_kg_theory_full is not None:
                ax_gg.plot(ell_theory, np.abs(np.array(cl_kg_theory_full)), ":", color="red",
                           alpha=0.4, lw=1.5, label=r"Limber |$C_\ell^{\kappa g}$| (full res.)")
            ax_gg.set_title(r"Galaxy Auto & Cross Spectra")
        else:
            ax_gg.set_title(r"Galaxy Auto Power Spectrum")

        ax_gg.set_xlabel(r"$\ell$")
        ax_gg.set_ylabel(r"$C_\ell$")
        ax_gg.legend(fontsize=9)
        ax_gg.grid(True, alpha=0.2)
        ax_gg.set_xlim(10, NYQUIST_FRACTION * ell_nyquist)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Info box
    box_x, box_y, box_z = (float(model.box_shape[i]) for i in range(3))
    cell_val = float(model.cell_shape[0]) if hasattr(model, "cell_shape") else 0
    info = f"Box: [{box_x:.0f}, {box_y:.0f}, {box_z:.0f}] Mpc/h | Cell: {cell_val:.1f} Mpc/h"
    if cmb_enabled:
        info += f" | Field: {field_size:.1f}°"
    info += f" | N={n_real}"
    plt.figtext(0.5, 0.02, info, ha="center", fontsize=8,
                bbox={"facecolor": "oldlace", "alpha": 0.5,
                      "edgecolor": "grey", "boxstyle": "round,pad=0.5"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"cl_spectra_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {outfile}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
