
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import cmb_lensing, metrics, plot
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    NYQUIST_FRACTION,
    compute_cl_high_z,
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kg,
)
from desi_cmb_fli.metrics import bin_cl_log


def plot_field_slices(truth, output_dir, mesh_shape=None, show=False, box_shape=None, field_size_deg=None, field_npix=None, chi_center=None, observation_mode="closure"):
    """
    Plot 2D slices of the generated fields (obs, kappa_obs, kappa_pred).

    Args:
        truth (dict): Dictionary containing 'obs', and optionally 'kappa_obs', 'kappa_pred'.
        output_dir (Path or str): Output directory.
        mesh_shape (tuple): Mesh shape (optional, inferred from obs).
        show (bool): Whether to show the plot.
        box_shape (tuple): Box size in Mpc/h (for galaxy projection).
        field_size_deg (float): Field size in degrees (for galaxy projection).
        field_npix (int): Number of pixels for projection.
        chi_center (float): Comoving distance to box center (for galaxy projection).
        observation_mode (str): 'closure' or 'abacus' (adjusts panel titles).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Galaxy Field Slices (XY, XZ, YZ) - only if obs is available
    if "obs" in truth:
        if mesh_shape is None:
            mesh_shape = truth["obs"].shape

        idx = mesh_shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # axis=0 -> YZ (x fixed), axis=1 -> XZ (y fixed), axis=2 -> XY (z fixed)
        titles = [f"YZ (x={idx})", f"XZ (y={idx})", f"XY (z={idx})"]
        axes_idx = [0, 1, 2]

        for ax, axis_idx, title in zip(axes, axes_idx, titles, strict=False):
            plt.sca(ax)
            plot.plot_mesh(truth["obs"], sli=slice(idx, idx + 1), axis=axis_idx, cmap="viridis")
            ax.set_title(title)

        plt.tight_layout()
        plt.savefig(output_dir / "obs_slices.png", dpi=150)
        if show:
            plt.show()
        plt.close()
    elif mesh_shape is None and "kappa_obs" in truth:
        # Fallback: use kappa shape if obs not available
        mesh_shape = (truth["kappa_obs"].shape[0], truth["kappa_obs"].shape[1], truth["kappa_obs"].shape[1])

    # 2. CMB Convergence (if available)
    if "kappa_obs" in truth:
        # Determine number of panels based on available data
        has_galaxy = "obs" in truth
        has_kappa_pred = "kappa_pred" in truth

        if has_galaxy:
            # Full mode: 3 panels (kappa_obs, kappa_pred, galaxy)
            ncols = 3
            figsize = (18, 5)
        else:
            # CMB-only: 2 panels (kappa_obs, kappa_pred)
            ncols = 2
            figsize = (12, 5)

        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 2:
            axes = list(axes)

        # Observed kappa
        _obs_arr = np.asarray(truth["kappa_obs"])
        _vmax_obs = float(np.percentile(np.abs(_obs_arr), 99))
        _obs_title = ("κ observed (Abacus + N_ℓ noise)" if observation_mode == "abacus"
                      else "CMB Convergence κ (observed)")
        im0 = axes[0].imshow(_obs_arr, origin="lower", cmap="RdBu_r",
                             vmin=-_vmax_obs, vmax=_vmax_obs)
        axes[0].set_title(_obs_title)
        axes[0].set_xlabel("x [pix]")
        axes[0].set_ylabel("y [pix]")
        plt.colorbar(im0, ax=axes[0], label="κ")

        # Predicted kappa
        _pred_title = ("κ Abacus (noiseless)" if observation_mode == "abacus"
                       else "CMB Convergence κ (predicted)")
        if has_kappa_pred:
            _pred_arr = np.asarray(truth["kappa_pred"])
            _vmax_pred = float(np.percentile(np.abs(_pred_arr), 99))
            im1 = axes[1].imshow(_pred_arr, origin="lower", cmap="RdBu_r",
                                 vmin=-_vmax_pred, vmax=_vmax_pred)
            axes[1].set_title(_pred_title)
            plt.colorbar(im1, ax=axes[1], label="κ")
        else:
            axes[1].axis('off')

        # Galaxy projected (only if available and 3-panel mode)
        if has_galaxy:
            if box_shape is not None and field_size_deg is not None and field_npix is not None and chi_center is not None:
                gxy_field = np.array(truth["obs"])
                gxy_proj = cmb_lensing.project_flat_sky(
                    gxy_field,
                    box_shape,
                    field_size_deg,
                    field_npix,
                    chi_center
                )
                im2 = axes[2].imshow(gxy_proj, origin="lower", cmap="viridis")
                axes[2].set_title("Galaxy Density (projected)")
                axes[2].set_xlabel("x [pix]")
                axes[2].set_ylabel("y [pix]")
                plt.colorbar(im2, ax=axes[2], label="N [counts]")
            else:
                # Fallback to slice if projection parameters not provided
                idx = truth["obs"].shape[2] // 2 if mesh_shape is None else mesh_shape[2] // 2
                im2 = axes[2].imshow(truth["obs"][..., idx], origin="lower", cmap="viridis")
                axes[2].set_title(f"Galaxy Density Slice (z={idx})")
                plt.colorbar(im2, ax=axes[2], label="δ")

        plt.tight_layout()
        plt.savefig(output_dir / "kappa_maps.png", dpi=150)
        if show:
            plt.show()
        plt.close()


def measure_spectra(truth, model, model_config=None):
    """
    Measure Cℓ spectra on the maps in `truth`.

    Pure measurement — no plotting, no theory curves.
    Use this to accumulate spectra over multiple realizations.

    Args:
        truth (dict): Map dictionary with keys like 'kappa_pred', 'kappa_obs', 'obs'.
        model (FieldLevelModel): Initialized model (for field geometry).
        model_config (dict, optional): Model config dict (for box_shape in galaxy projection).

    Returns:
        dict: {ell, cl_kk_pred, cl_kk_obs, cl_gg, cl_kg}. Missing keys are None.
    """
    if model_config is None:
        model_config = getattr(model, "config", {})

    cmb_enabled = model.cmb_enabled
    has_kappa_pred = "kappa_pred" in truth
    has_kappa_obs = "kappa_obs" in truth
    has_galaxies = "obs" in truth

    if cmb_enabled:
        field_size = model.cmb_field_size_deg
        npix = model.cmb_field_npix
    else:
        chi_max_box = model.box_shape[2]
        half_angle_rad = np.arctan(model.box_shape[0] / (2.0 * chi_max_box))
        field_size = float(2.0 * half_angle_rad * (180.0 / np.pi))
        npix = 256

    chi_center_val = float(model.box_center[2]) if hasattr(model, "box_center") else float(model.box_shape[2] / 2.0)

    ell = None
    cl_kk_pred, cl_kk_obs, cl_gg, cl_kg = None, None, None, None

    # Kappa spectra
    if cmb_enabled and has_kappa_obs:
        ell, cl_kk_obs = metrics.get_cl_2d(truth["kappa_obs"], field_size_deg=field_size)
        ell, cl_kk_obs = np.asarray(ell), np.asarray(cl_kk_obs)

    if cmb_enabled and has_kappa_pred:
        ell_tmp, cl_kk_pred = metrics.get_cl_2d(truth["kappa_pred"], field_size_deg=field_size)
        cl_kk_pred = np.asarray(cl_kk_pred)
        if ell is None:
            ell = np.asarray(ell_tmp)

    # Galaxy spectra
    if has_galaxies:
        box_shape = model_config.get("box_shape", model.box_shape)
        gxy_field = np.array(truth["obs"])
        gxy_proj = cmb_lensing.project_flat_sky(
            gxy_field, box_shape, field_size,
            npix if cmb_enabled else 256, chi_center_val,
        )
        gxy_mean = jnp.mean(gxy_proj)
        gxy_delta = (gxy_proj - gxy_mean) / gxy_mean
        ell_tmp, cl_gg = metrics.get_cl_2d(gxy_delta, field_size_deg=field_size)
        cl_gg = np.asarray(cl_gg)
        if ell is None:
            ell = np.asarray(ell_tmp)

        # Cross-spectrum κ × g
        if cmb_enabled and has_kappa_pred:
            _, cl_kg = metrics.get_cl_2d(truth["kappa_pred"], gxy_delta, field_size_deg=field_size)
            cl_kg = np.asarray(cl_kg)

    # ── Log-binned versions ────────────────────────────────────────────────
    ell_b, cl_kk_pred_b, cl_kk_obs_b, cl_gg_b, cl_kg_b = (None,) * 5
    n_modes_b = None
    if ell is not None:
        if cl_kk_obs is not None:
            ell_b, cl_kk_obs_b, n_modes_b = bin_cl_log(ell, cl_kk_obs)
        if cl_kk_pred is not None:
            ell_b_tmp, cl_kk_pred_b, n_modes_b_tmp = bin_cl_log(ell, cl_kk_pred)
            if ell_b is None:
                ell_b, n_modes_b = ell_b_tmp, n_modes_b_tmp
        if cl_gg is not None:
            ell_b_gg, cl_gg_b, _ = bin_cl_log(ell, cl_gg)
            if ell_b is None:
                ell_b = ell_b_gg
        if cl_kg is not None:
            _, cl_kg_b, _ = bin_cl_log(ell, cl_kg)

    return {"ell": ell, "cl_kk_pred": cl_kk_pred, "cl_kk_obs": cl_kk_obs,
            "cl_gg": cl_gg, "cl_kg": cl_kg,
            # binned versions
            "ell_b": ell_b, "cl_kk_pred_b": cl_kk_pred_b, "cl_kk_obs_b": cl_kk_obs_b,
            "cl_gg_b": cl_gg_b, "cl_kg_b": cl_kg_b, "n_modes_b": n_modes_b}


def plot_spectra(truth, model, output_dir, cosmo_params=None, model_config=None,
                 observation_mode="closure", show=False, suffix=""):
    """
    Measure and plot Cℓ spectra from the maps in `truth`, compared to Limber theory.

    Works for any observation mode (closure, Abacus, real data).
    Measures spectra on the provided maps — does NOT generate new realizations.

    Panels:
        - κκ auto-spectrum:
            closure: 3 pairs (grey=obs, purple=pred+highz, blue=pred)
            abacus:  2 pairs (grey=obs, purple=pred which is full LOS)
        - Galaxy auto + κg cross-spectrum (if obs present)

    Args:
        truth (dict): Map dictionary. Recognized keys:
            'kappa_pred' (2D, noiseless κ), 'kappa_obs' (2D, with noise),
            'obs' (3D galaxy field), 'matter_mesh' (3D matter field).
        model (FieldLevelModel): Initialized model (for field geometry, N_ell, etc.).
        output_dir (Path or str): Output directory for the plot.
        cosmo_params (dict, optional): Cosmology for Limber curves {Omega_m, sigma8, ...}.
            If None, uses model.loc_fid.
        model_config (dict, optional): Model config dict (for box_shape in galaxy projection).
        observation_mode (str): 'closure' or 'abacus'. Controls which curves are shown.
        show (bool): Whether to show the plot interactively.
        suffix (str): Suffix for the output filename.

    Returns:
        dict: Measured spectra {ell, cl_kk_pred, cl_kk_obs, cl_gg, cl_kg}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_config is None:
        model_config = getattr(model, "config", {})

    print("=" * 80)
    print("VALIDATION: Power Spectra")
    print("=" * 80)

    cmb_enabled = model.cmb_enabled
    has_kappa_obs = "kappa_obs" in truth
    has_galaxies = "obs" in truth

    if not has_kappa_obs and not has_galaxies:
        print("  No maps to measure spectra on, skipping.")
        return {}

    # ── Cosmology for Limber theory ──────────────────────────────────────────
    if cosmo_params is None:
        cosmo_params = {k: v for k, v in model.loc_fid.items()
                        if k in ("Omega_m", "sigma8")}
    cosmo_val = get_cosmology(**cosmo_params)

    # ── Field geometry ───────────────────────────────────────────────────────
    if cmb_enabled:
        field_size = model.cmb_field_size_deg
        npix = model.cmb_field_npix
    else:
        chi_max_box = model.box_shape[2]
        half_angle_rad = np.arctan(model.box_shape[0] / (2.0 * chi_max_box))
        field_size = float(2.0 * half_angle_rad * (180.0 / np.pi))
        npix = 256

    ell_nyquist = np.pi * npix / (field_size * np.pi / 180.0)

    # ── Theoretical Limber spectra ───────────────────────────────────────────
    z_source = model.cmb_z_source
    chi_min, chi_max_theory = 1.0, float(model.box_shape[2])

    b1_lag = cosmo_params.get("b1", model.loc_fid.get("b1", 1.0))
    bE = 1.0 + b1_lag

    ell_theory = np.geomspace(10, 2000, 100)

    cl_kk_theory, cl_gg_theory, cl_kg_theory = None, None, None
    nell_at_theory = np.zeros_like(ell_theory)
    cl_high_z_theory = np.zeros_like(ell_theory)

    if cmb_enabled:
        # Box-only kappa theory (chi_min to chi_max_box)
        cl_kk_theory = np.asarray(compute_theoretical_cl_kappa(
            cosmo_val, jnp.array(ell_theory), chi_min, chi_max_theory, z_source
        ))

        # N_ell interpolated to theory grid
        ell_grid_flat = model.ell_grid.flatten()
        sort_idx = np.argsort(ell_grid_flat)
        nell_at_theory = np.exp(np.interp(
            np.log(np.maximum(ell_theory, 1e-5)),
            np.log(np.maximum(ell_grid_flat[sort_idx], 1e-5)),
            np.log(model.nell_grid.flatten()[sort_idx])
        ))

        # High-z correction
        if model.full_los_correction:
            _high_z_mode = 'exact' if observation_mode == 'abacus' else model.high_z_mode
            cl_high_z_grid = compute_cl_high_z(
                cosmo_val,
                model.ell_grid,
                model.box_shape[2],
                model.chi_high_z_max,
                z_source,
                mode=_high_z_mode, cl_cached=model.cl_high_z_cached,
                gradients=model.high_z_gradients, loc_fid=model.loc_fid,
            )
            cl_high_z_flat = np.array(cl_high_z_grid).flatten()
            cl_high_z_theory = np.interp(ell_theory, ell_grid_flat[sort_idx], cl_high_z_flat[sort_idx])

    if has_galaxies:
        cl_gg_theory = np.asarray(compute_theoretical_cl_gg(
            cosmo_val, jnp.array(ell_theory), chi_min, chi_max_theory, bE
        ))
        if cmb_enabled:
            cl_kg_theory = np.asarray(compute_theoretical_cl_kg(
                cosmo_val, jnp.array(ell_theory), chi_min, chi_max_theory, z_source, bE
            ))

    # ── Measure spectra on the provided maps ─────────────────────────────────
    spectra = measure_spectra(truth, model, model_config)
    ell        = spectra["ell"]
    cl_kk_pred = spectra["cl_kk_pred"]
    cl_kk_obs  = spectra["cl_kk_obs"]
    cl_gg      = spectra["cl_gg"]
    cl_kg      = spectra["cl_kg"]
    # Binned versions
    ell_b        = spectra["ell_b"]
    cl_kk_pred_b = spectra["cl_kk_pred_b"]
    cl_kk_obs_b  = spectra["cl_kk_obs_b"]
    cl_gg_b      = spectra["cl_gg_b"]
    cl_kg_b      = spectra["cl_kg_b"]
    valid_b = ell_b is not None and len(ell_b) > 0

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("  Generating plot...")
    has_cmb_panel = cmb_enabled and (cl_kk_pred is not None or cl_kk_obs is not None)
    has_gxy_panel = cl_gg is not None

    if has_cmb_panel and has_gxy_panel:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_kk, ax_gg = axes
    elif has_cmb_panel:
        fig, ax_kk = plt.subplots(1, 1, figsize=(8, 6))
        ax_gg = None
    elif has_gxy_panel:
        fig, ax_gg = plt.subplots(1, 1, figsize=(8, 6))
        ax_kk = None
    else:
        print("  Nothing to plot.")
        return {}

    # ── CMB κ panel ──────────────────────────────────────────────────────────
    if has_cmb_panel:
        plt.sca(ax_kk)
        valid = (ell > 10) & (ell < NYQUIST_FRACTION * ell_nyquist)

        cl_box_highz_theory = cl_kk_theory + cl_high_z_theory
        cl_total_theory = cl_box_highz_theory + nell_at_theory

        cl_high_z_on_ell = np.interp(ell, ell_theory, cl_high_z_theory)

        # ── Raw faded lines ───────────────────────────────────────────────────
        if cl_kk_obs is not None:
            v_grey = valid & np.isfinite(cl_kk_obs)
            plt.loglog(ell[v_grey], cl_kk_obs[v_grey], "-", color="grey", lw=0.4, alpha=0.3)
        if cl_kk_pred is not None:
            _pred_raw = (cl_kk_pred + cl_high_z_on_ell) if observation_mode == "closure" else cl_kk_pred
            v_pred_raw = valid & np.isfinite(_pred_raw)
            _pred_raw_color = "steelblue" if observation_mode == "closure" else "purple"
            plt.loglog(ell[v_pred_raw], _pred_raw[v_pred_raw], "-", color=_pred_raw_color, lw=0.4, alpha=0.3)

        # ── Binned markers ────────────────────────────────────────────────────
        if valid_b:
            vb_mask = (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist)
            if cl_kk_obs_b is not None:
                vb = vb_mask & np.isfinite(cl_kk_obs_b)
                plt.errorbar(ell_b[vb], cl_kk_obs_b[vb], fmt="o", color="grey",
                             ms=4, lw=1.2, capsize=2,
                             label=r"$C_\ell^{\kappa\kappa}$ (obs, binned)")
            if cl_kk_pred_b is not None:
                _pred_b = (cl_kk_pred_b + np.interp(ell_b, ell_theory, cl_high_z_theory)
                           if observation_mode == "closure" else cl_kk_pred_b)
                vb = vb_mask & np.isfinite(_pred_b)
                _pred_b_label = (
                    r"$C_\ell^{\kappa\kappa}$ (pred + high-$z$, binned)"
                    if observation_mode == "closure"
                    else r"$C_\ell^{\kappa\kappa}$ (Abacus, noiseless, binned)"
                )
                plt.errorbar(ell_b[vb], _pred_b[vb], fmt="s", color="purple",
                             ms=4, lw=1.2, capsize=2,
                             label=_pred_b_label)
            if observation_mode == "closure" and cl_kk_pred_b is not None:
                vb = vb_mask & np.isfinite(cl_kk_pred_b)
                plt.errorbar(ell_b[vb], cl_kk_pred_b[vb], fmt="^", color="steelblue",
                             ms=4, lw=1, capsize=2,
                             label=r"$C_\ell^{\kappa\kappa}$ (pred box, binned)")

        # ── Theory dashed lines ───────────────────────────────────────────────
        plt.plot(ell_theory, cl_total_theory, "--", color="grey", lw=1.5, alpha=0.9,
                 label=r"Limber (Box + high-$z$ + $N_\ell$)")
        plt.plot(ell_theory, cl_box_highz_theory, "--", color="purple", lw=1.5, alpha=0.9,
                 label=r"Limber (Box + high-$z$)")
        if observation_mode == "closure":
            plt.plot(ell_theory, cl_kk_theory, "--", color="steelblue", lw=1.5, alpha=0.9,
                     label=r"Limber (Box)")

        mode_label = "closure" if observation_mode == "closure" else "Abacus"
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.title(rf"$C_\ell^{{\kappa\kappa}}$ — {mode_label}")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.2)
        plt.xlim(10, NYQUIST_FRACTION * ell_nyquist)

    # ── Galaxy panel ─────────────────────────────────────────────────────────
    if has_gxy_panel:
        plt.sca(ax_gg)
        valid_gg = (ell > 10) & (ell < NYQUIST_FRACTION * ell_nyquist) & np.isfinite(cl_gg)
        plt.loglog(ell[valid_gg], cl_gg[valid_gg], "-", color="green", lw=0.4, alpha=0.3)
        if valid_b and cl_gg_b is not None:
            vb_gg = (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist) & np.isfinite(cl_gg_b)
            plt.errorbar(ell_b[vb_gg], cl_gg_b[vb_gg], fmt="o", color="green",
                         ms=4, lw=1.2, capsize=2, label=r"$C_\ell^{gg}$ (binned)")
        if cl_gg_theory is not None:
            plt.plot(ell_theory, np.array(cl_gg_theory), "--", color="green",
                     alpha=0.8, lw=2, label=r"Limber $C_\ell^{gg}$")

        if cl_kg is not None:
            valid_kg = (ell > 10) & (ell < NYQUIST_FRACTION * ell_nyquist) & np.isfinite(cl_kg)
            plt.loglog(ell[valid_kg], np.abs(cl_kg[valid_kg]), "-", color="red",
                       lw=0.4, alpha=0.3)
            if valid_b and cl_kg_b is not None:
                vb_kg = (ell_b > 10) & (ell_b < NYQUIST_FRACTION * ell_nyquist) & np.isfinite(cl_kg_b)
                plt.errorbar(ell_b[vb_kg], np.abs(cl_kg_b[vb_kg]), fmt="s", color="red",
                             ms=4, lw=1.2, capsize=2, label=r"|$C_\ell^{\kappa g}$| (binned)")
            if cl_kg_theory is not None:
                plt.plot(ell_theory, np.abs(np.array(cl_kg_theory)), "--", color="red",
                         alpha=0.8, lw=2, label=r"Limber |$C_\ell^{\kappa g}$|")
            plt.title(r"Galaxy Auto & Cross Spectra")
        else:
            plt.title(r"Galaxy Auto Power Spectrum")

        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.xlim(10, NYQUIST_FRACTION * ell_nyquist)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Info box
    box_x, box_y, box_z = (float(model.box_shape[i]) for i in range(3))
    mesh_x, mesh_y, mesh_z = (int(model.mesh_shape[i]) for i in range(3))
    cell_val = float(model.cell_shape[0]) if hasattr(model, "cell_shape") else 0
    info = (
        f"Box: [{box_x:.0f}, {box_y:.0f}, {box_z:.0f}] Mpc/h | "
        f"Mesh: ({mesh_x}, {mesh_y}, {mesh_z}) | Cell: {cell_val:.1f} Mpc/h"
    )
    if cmb_enabled:
        info += f" | Field: {field_size:.1f}°"
    if cosmo_params:
        info += f" | Ω_m={cosmo_params.get('Omega_m', '?')}, σ₈={cosmo_params.get('sigma8', '?')}"
    plt.figtext(0.5, 0.02, info, ha="center", fontsize=8,
                bbox={"facecolor": "oldlace", "alpha": 0.5,
                      "edgecolor": "grey", "boxstyle": "round,pad=0.5"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"cl_spectra{suffix}_{timestamp}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {outfile}")

    if show:
        plt.show()
    plt.close()

    return {"ell": ell, "cl_kk_pred": cl_kk_pred, "cl_kk_obs": cl_kk_obs,
            "cl_gg": cl_gg, "cl_kg": cl_kg}


def plot_cmb_noise_spectrum(model, output_dir, show=False):
    """
    Plot the input CMB lensing noise spectrum N_ell.

    Args:
        model (FieldLevelModel): Initialized model containing cmb_noise_nell.
        output_dir (Path or str): Output directory.
        show (bool): Whether to show the plot.
    """
    if not (model.cmb_enabled and model.cmb_noise_nell is not None):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("VALIDATION: Plotting CMB noise spectrum...")
    try:
        # Resolve N_ell
        if isinstance(model.cmb_noise_nell, str):
            data = np.loadtxt(model.cmb_noise_nell)
            if data.ndim == 1:
                ell_in = np.arange(len(data))
                nell_in = data
            else:
                ell_in, nell_in = data[:, 0], data[:, 1]
        elif isinstance(model.cmb_noise_nell, dict):
            ell_in, nell_in = model.cmb_noise_nell["ell"], model.cmb_noise_nell["N_ell"]
        else:
            # Tuple or list
            ell_in, nell_in = model.cmb_noise_nell[0], model.cmb_noise_nell[1]

        # Apply scaling if present
        nell_scaled = nell_in * model.cmb_noise_scaling

        # Filter for log plot
        mask = (ell_in > 0) & (nell_scaled > 0)

        plt.figure(figsize=(8, 6))
        plot.plot_cl(ell_in[mask], nell_scaled[mask], log=True, ylabel=r"$N_\ell$")

        # Title reflects scaling
        if model.cmb_noise_scaling != 1.0:
            plt.title(f"CMB Lensing Noise Power Spectrum (scaled by {model.cmb_noise_scaling:.4g})")
        else:
            plt.title("CMB Lensing Noise Power Spectrum")

        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(["$N_\\ell$ (used in run)"])

        outfile = output_dir / "cmb_noise_spectrum.png"
        plt.savefig(outfile, dpi=150)
        if show:
            plt.show()
        plt.close()
        print(f"✓ Saved: {outfile}")
    except Exception as e:
        print(f"⚠️  Could not plot N_ell: {e}")

def plot_warmup_diagnostics(model, state, init_params, truth, output_dir, show=False):
    """
    Plot warmup diagnostics: Power Spectrum, Transfer Function, and Coherence
    comparing initial condition (init) vs warmed-up state (warm).

    Args:
        model (FieldLevelModel): The model.
        state (MCMCState): The final warmup state.
        init_params (dict): The initial parameters before warmup (all chains).
        truth (dict): The truth dictionary containing 'init_mesh'.
        output_dir (Path or str): Output directory.
        show (bool): Whether to show the plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("VALIDATION: Warmup Diagnostics (Power/Transfer/Coherence)")

    try:
        # Extract true init_mesh from truth
        init_mesh_true = truth.get('init_mesh')

        if init_mesh_true is not None:
            # Convert init_mesh from Fourier to real space for spectrum calculation
            init_mesh_true_real = jnp.fft.irfftn(init_mesh_true)

            # Compute true spectrum
            kpow_true = model.spectrum(init_mesh_true_real)

            # Compute power/transfer/coherence for init params (all chains)
            # Need to extract init_mesh from reparametrized params and convert to real
            def compute_kptc(params_dict):
                # Reparam to get init_mesh in Fourier (base) space
                params_base = model.reparam(params_dict)
                # Convert from Fourier to real space
                init_mesh_real = jnp.fft.irfftn(params_base['init_mesh'])
                return model.powtranscoh(init_mesh_true_real, init_mesh_real)

            # Vectorize over chains
            from jax import vmap
            kptcs_init = vmap(compute_kptc)(init_params)
            kptcs_warm = vmap(compute_kptc)(state.position)

            # Fiducial linear power spectrum
            cosmo_fid = get_cosmology(**model.loc_fid)
            # Use k bins from first chain result
            from desi_cmb_fli.bricks import lin_power_interp
            kptcs_warm_0 = jax.tree.map(lambda x: x[0], kptcs_warm)
            kpow_fid = (kptcs_warm_0[0], lin_power_interp(cosmo_fid)(kptcs_warm_0[0]))

            # Create diagnostic figure
            fig = plt.figure(figsize=(12, 4))
            fig.suptitle('Warmup Diagnostics: Initial Conditions Spectrum', fontsize=14)

            def plot_kptcs(kptcs, label=None):
                """Plot power/transfer/coherence (median of chains)."""
                kptcs_median = jax.tree.map(lambda x: jnp.median(x, 0), kptcs)
                plot.plot_powtranscoh(*kptcs_median, label=label)

            # Plot init and warmup
            plot_kptcs(kptcs_init, label='init')
            plot_kptcs(kptcs_warm, label='warm')

            # Add truth and fiducial to subplot 1 (power)
            plt.subplot(131)
            plot.plot_pow(*kpow_true, 'k:', label='true')
            plot.plot_pow(*kpow_fid, 'k--', alpha=0.5, label='fiducial')
            plt.legend()

            # Add reference lines to subplot 2 (transfer)
            plt.subplot(132)
            plt.axhline(1., linestyle=':', color='k', alpha=0.5)
            # Fiducial transfer (sqrt of power ratio)
            k_true, pow_true = kpow_true
            k_fid, pow_fid = kpow_fid
            transfer_fid = (pow_fid / pow_true)**0.5
            plot.plot_trans(k_true, transfer_fid, 'k--', alpha=0.5, label='fiducial')

            # Add reference line to subplot 3 (coherence)
            plt.subplot(133)
            # Plot mean of selection mask if available
            if hasattr(model, 'selec_mesh'):
                selec_mean = float(jnp.mean(model.selec_mesh))
                plt.axhline(selec_mean, linestyle=':', color='k', alpha=0.5,
                           label=f'selec_mesh mean={selec_mean:.3f}')
            plt.axhline(1.0, linestyle=':', color='k', alpha=0.2)

            plt.tight_layout()
            outfile = output_dir / 'init_warm.png'
            plt.savefig(outfile, dpi=150)
            if show:
                plt.show()
            plt.close()
            print(f"✓ Saved: {outfile}")
        else:
            print("⚠️  Warning: init_mesh not found in truth, skipping warmup diagnostic plot")

    except Exception as e:
        print(f"⚠️  Warning: Could not generate warmup diagnostic plot: {e}")
