
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli import cmb_lensing, metrics, plot
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_cl_high_z,
    compute_theoretical_cl_gg,
    compute_theoretical_cl_kappa,
    compute_theoretical_cl_kg,
)


def plot_field_slices(truth, output_dir, mesh_shape=None, show=False, box_shape=None, field_size_deg=None, field_npix=None, chi_center=None):
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
        im0 = axes[0].imshow(truth["kappa_obs"], origin="lower", cmap="RdBu_r")
        axes[0].set_title("CMB Convergence κ (observed)")
        axes[0].set_xlabel("x [pix]")
        axes[0].set_ylabel("y [pix]")
        plt.colorbar(im0, ax=axes[0], label="κ")

        # Predicted kappa (if available)
        if has_kappa_pred:
            im1 = axes[1].imshow(truth["kappa_pred"], origin="lower", cmap="RdBu_r")
            axes[1].set_title("CMB Convergence κ (predicted)")
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

def compute_and_plot_spectra(model, truth_params, output_dir=None, n_realizations=1, seed=None, show=False, suffix="", model_config=None):
    """
    Compute and plot C_l spectra (kappa-kappa, galaxy-galaxy, cross) from simulations.
    Can run multiple realizations to estimate variance.

    Args:
        model (FieldLevelModel): Initialized model.
        truth_params (dict): Truth parameters for generation.
        output_dir (str or Path, optional): Output directory. Defaults to 'figures'.
        n_realizations (int): Number of realizations to average.
        seed (int, optional): Base random seed.
        show (bool): Whether to show the plot interactively.
        suffix (str): Suffix for the output filename.
        model_config (dict): Model configuration (optional, if not provided will try to use model.config).

    Returns:
        dict: Dictionary containing computed spectra (mean and std).
    """
    if output_dir is None:
        output_dir = Path("figures")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VALIDATION: Computing & Plotting Spectra")
    print("=" * 80)

    if model_config is None:
        # Fallback if model has config attribute (future proofing)
        model_config = getattr(model, "config", {})

    cmb_enabled = model.cmb_enabled
    print(f"CMB Lensing Enabled: {cmb_enabled}")

    if cmb_enabled:
        print(f"  Sky area: {model.cmb_field_size_deg**2:.2f} deg²")
        print(f"  CMB npix: {model.cmb_field_npix}")

    base_seed = seed if seed is not None else 42 # Default seed if None

    # Get cosmology for projection
    cosmo_val = get_cosmology(**truth_params)
    chi_center_val = float(model.box_center[2]) if hasattr(model, "box_center") else float(model.box_shape[2] / 2.0)

    # Determine field size for projection
    if cmb_enabled:
        field_size = model.cmb_field_size_deg
        npix = model.cmb_field_npix
    else:
        # Calculate field size for full box at z_center
        chi_max_box = model.box_shape[2]
        half_angle_rad = np.arctan(model.box_shape[0] / (2.0 * chi_max_box))
        field_size = float(2.0 * half_angle_rad * (180.0 / np.pi))
        npix = 256 # Default npix for visualization
        print(f"  Computed sky area (for galaxy projection): {field_size**2:.2f} deg²")

    # Storage
    all_cl_kk_box = []
    all_cl_kk_obs = []
    all_cl_gg = []
    all_cl_kg = []
    ell = None

    # Loop over realizations
    print(f"Generating {n_realizations} realization(s)...")
    for i_real in range(n_realizations):
        seed_i = base_seed + i_real
        print(f"  Realization {i_real+1}/{n_realizations} (seed={seed_i})", end="\r")

        # Generate truth
        truth = model.predict(
            samples=truth_params,
            hide_base=False,
            hide_samp=False,
            hide_det=False,
            frombase=True,
            rng=jr.key(seed_i),
        )

        # Project galaxy field (only if galaxies enabled)
        if "obs" in truth:
            gxy_field = np.array(truth["obs"])
            gxy_proj = cmb_lensing.project_flat_sky(
                gxy_field,
                model_config["box_shape"],
                field_size,
                npix if cmb_enabled else 256,
                chi_center_val
            )

            # Compute spectra
            gxy_mean = jnp.mean(gxy_proj)
            gxy_delta = (gxy_proj - gxy_mean) / gxy_mean
            ell_i, cl_gg_i = metrics.get_cl_2d(gxy_delta, field_size_deg=field_size)
            all_cl_gg.append(np.array(cl_gg_i))

        if cmb_enabled:
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

            ell_i, cl_kk_box_i = metrics.get_cl_2d(kappa_box, field_size_deg=field_size)
            _, cl_kk_obs_i = metrics.get_cl_2d(kappa_obs, field_size_deg=field_size)

            # Compute cross-spectrum only if galaxy data available
            if "obs" in truth:
                _, cl_kg_i = metrics.get_cl_2d(kappa_box, gxy_delta, field_size_deg=field_size)
                all_cl_kg.append(np.array(cl_kg_i))

            all_cl_kk_box.append(np.array(cl_kk_box_i))
            all_cl_kk_obs.append(np.array(cl_kk_obs_i))
            if ell is None:
                ell = ell_i

        if ell is None and len(all_cl_gg) > 0:
            ell = ell_i
    print("") # Newline

    # Compute mean and std
    if len(all_cl_gg) > 0:
        all_cl_gg = np.array(all_cl_gg)
        cl_gg_mean = np.nanmean(all_cl_gg, axis=0)
        cl_gg_std = np.nanstd(all_cl_gg, axis=0)
    else:
        cl_gg_mean, cl_gg_std = None, None

    if cmb_enabled:
        if len(all_cl_kk_box) > 0:
            all_cl_kk_box = np.array(all_cl_kk_box)
            all_cl_kk_obs = np.array(all_cl_kk_obs)

            cl_kk_box_mean = np.nanmean(all_cl_kk_box, axis=0)
            cl_kk_box_std = np.nanstd(all_cl_kk_box, axis=0)
            cl_kk_obs_mean = np.nanmean(all_cl_kk_obs, axis=0)
        else:
            cl_kk_box_mean, cl_kk_obs_mean = None, None
            cl_kk_box_std = None

        if len(all_cl_kg) > 0:
            all_cl_kg = np.array(all_cl_kg)
            cl_kg_mean = np.nanmean(all_cl_kg, axis=0)
            cl_kg_std = np.nanstd(all_cl_kg, axis=0)
        else:
            cl_kg_mean, cl_kg_std = None, None
    else:
        cl_kk_box_mean, cl_kk_obs_mean, cl_kg_mean = None, None, None

    # Theoretical Spectra (Limber)
    print("Computing theoretical spectra...")
    chi_min, chi_max = 1.0, float(model.box_shape[2])
    z_source = model.cmb_z_source
    b1_lag = truth_params.get("b1", 1.0)
    bE = 1.0 + b1_lag

    ell_clean = np.geomspace(10, 2000, 100)
    cl_gg_theory_clean = compute_theoretical_cl_gg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, bE)

    if cmb_enabled:
        cl_kk_theory_clean = compute_theoretical_cl_kappa(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source)
        cl_kg_theory_clean = compute_theoretical_cl_kg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source, bE)

        # Compute high-z correction using centralized function
        cl_high_z_theory = np.zeros_like(ell_clean)
        if model.full_los_correction:
            # Compute high-z on native grid first
            cl_high_z_grid = compute_cl_high_z(
                cosmo_val,
                model.ell_grid,
                model.box_shape[2],  # chi_min
                None,  # chi_max computed internally
                z_source,
                mode=model.high_z_mode,
                cl_cached=model.cl_high_z_cached,
                gradients=model.high_z_gradients,
                loc_fid=model.loc_fid
            )

            # Interpolate to ell_clean for plotting
            ell_grid_flat = model.ell_grid.flatten()
            cl_high_z_flat = np.array(cl_high_z_grid).flatten()
            sort_idx = np.argsort(ell_grid_flat)
            ell_sorted = ell_grid_flat[sort_idx]
            cl_sorted = cl_high_z_flat[sort_idx]
            cl_high_z_theory = np.interp(ell_clean, ell_sorted, cl_sorted)

        print(f"  High-z correction mode: {model.high_z_mode if model.full_los_correction else 'disabled'}")

    # Plotting
    print("Generating plots...")

    # Compute Nyquist frequency in angular space: ell_max = pi * npix / theta_field_rad
    ell_nyquist = np.pi * npix / (field_size * np.pi / 180.0)

    galaxies_enabled = model.galaxies_enabled if hasattr(model, 'galaxies_enabled') else (cl_gg_mean is not None)

    if cmb_enabled and galaxies_enabled:
        # Joint mode: show both CMB and galaxy panels
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_kk = axes[0]
        ax_gg = axes[1]
    elif cmb_enabled:
        # CMB-only mode: show only CMB panel
        fig, ax_kk = plt.subplots(1, 1, figsize=(8, 6))
        ax_gg = None
    else:
        # Galaxies-only mode: show only galaxy panel
        fig, ax_gg = plt.subplots(1, 1, figsize=(8, 6))
        ax_kk = None

    # CMB Kappa Auto Spectrum
    if cmb_enabled and ax_kk is not None:
        # Noise
        cmb_nell_cfg = model_config.get("cmb_noise_nell", {})
        nell_at_ell_clean = np.zeros_like(ell_clean)
        if isinstance(cmb_nell_cfg, str):
            try:
                nell_data = np.loadtxt(cmb_nell_cfg)
                if nell_data.ndim == 1:
                    nell_ell, nell_val = np.arange(len(nell_data)), nell_data
                else:
                    nell_ell, nell_val = nell_data[:, 0], nell_data[:, 1]
                nell_at_ell_clean = np.interp(ell_clean, nell_ell, nell_val) * model.cmb_noise_scaling
            except Exception:
                pass
        elif isinstance(cmb_nell_cfg, dict):
             nell_at_ell_clean = np.interp(ell_clean, np.array(cmb_nell_cfg["ell"]), np.array(cmb_nell_cfg["N_ell"])) * model.cmb_noise_scaling

        # Kappa Auto
        plt.sca(ax_kk)
        valid_idx = np.isfinite(ell) & np.isfinite(cl_kk_box_mean) & (ell > 0)
        ell_valid = ell[valid_idx]

        # Always show box (no noise)
        if n_realizations > 1:
            plt.fill_between(ell_valid, (cl_kk_box_mean - cl_kk_box_std)[valid_idx], (cl_kk_box_mean + cl_kk_box_std)[valid_idx], color='blue', alpha=0.2)
        plt.loglog(ell_valid, cl_kk_box_mean[valid_idx], '-', color='blue', lw=1.5, label=r"$C_\ell^{\kappa \kappa}$ (box)")
        plt.plot(ell_clean, np.array(cl_kk_theory_clean), '--', color='blue', alpha=0.8, lw=2, label="Theory (box)")

        # With or without high-z correction
        if model.full_los_correction and np.any(cl_high_z_theory > 0):
            # Compute high-z correction and interpolate to ell_valid for obs curve
            cl_high_z_grid = compute_cl_high_z(
                cosmo_val,
                model.ell_grid,
                model.box_shape[2],
                None,
                z_source,
                mode=model.high_z_mode,
                cl_cached=model.cl_high_z_cached,
                gradients=model.high_z_gradients,
                loc_fid=model.loc_fid
            )

            ell_grid_flat = model.ell_grid.flatten()
            cl_high_z_flat = np.array(cl_high_z_grid).flatten()
            sort_idx = np.argsort(ell_grid_flat)
            ell_sorted = ell_grid_flat[sort_idx]
            cl_sorted = cl_high_z_flat[sort_idx]

            # Measured: Box + N_ell + high-z
            cl_kk_total_obs = cl_kk_obs_mean[valid_idx]
            plt.loglog(ell_valid, cl_kk_total_obs, '-', color='red', lw=2,
                       label=r"Box + $N_\ell$ + $C_\ell^{\rm high-z}$")

            # Theory: Theory + N_ell + high-z
            cl_total_theory = np.array(cl_kk_theory_clean) + nell_at_ell_clean + cl_high_z_theory
            plt.plot(ell_clean, cl_total_theory, '--', color='red', alpha=0.8, lw=2,
                     label=r"Theory + $N_\ell$ + $C_\ell^{\rm high-z}$")
        else:
            # No high-z: just show Box + N_ell and Theory + N_ell
            plt.loglog(ell_valid, cl_kk_obs_mean[valid_idx], '-', color='gray', lw=2, label=r"Box + $N_\ell$")
            plt.plot(ell_clean, np.array(cl_kk_theory_clean) + nell_at_ell_clean, '--', color='gray', alpha=0.8, lw=2, label=r"Theory + $N_\ell$")

        plt.xlabel(r"$\ell$"), plt.ylabel(r"$C_\ell$"), plt.title(r"$C_\ell^{\kappa \kappa}$"), plt.legend(fontsize=9), plt.grid(True, alpha=0.2)
        plt.xlim(10, 0.85 * ell_nyquist)  # Cut at 0.85*Nyquist (inference limit)

    # Galaxy Auto & Cross Spectra
    if ax_gg is not None:
        plt.sca(ax_gg)
        if cl_gg_mean is not None:
            valid_idx_gg = np.isfinite(ell) & np.isfinite(cl_gg_mean) & (ell > 0)
            ell_valid_gg = ell[valid_idx_gg]
            if n_realizations > 1:
                plt.fill_between(ell_valid_gg, (cl_gg_mean - cl_gg_std)[valid_idx_gg], (cl_gg_mean + cl_gg_std)[valid_idx_gg], color='green', alpha=0.2)
            plt.loglog(ell_valid_gg, cl_gg_mean[valid_idx_gg], '-', color='green', lw=1.5, label=r"$C_\ell^{gg}$ (Mean)")
            plt.plot(ell_clean, np.array(cl_gg_theory_clean), '--', color='green', alpha=0.8, lw=2, label=r"Theory $C_\ell^{gg}$")

        if cmb_enabled and cl_kg_mean is not None:
            if n_realizations > 1:
                valid_idx_gg = np.isfinite(ell) & np.isfinite(cl_kg_mean) & (ell > 0)
                ell_valid_gg = ell[valid_idx_gg]
                plt.fill_between(ell_valid_gg, np.abs(cl_kg_mean - cl_kg_std)[valid_idx_gg], np.abs(cl_kg_mean + cl_kg_std)[valid_idx_gg], color='red', alpha=0.2)
            else:
                valid_idx_gg = np.isfinite(ell) & np.isfinite(cl_kg_mean) & (ell > 0)
                ell_valid_gg = ell[valid_idx_gg]
            plt.loglog(ell_valid_gg, np.abs(cl_kg_mean)[valid_idx_gg], '-', color='red', lw=1.5, label=r"|$C_\ell^{\kappa g}$| (Mean)")
            plt.plot(ell_clean, np.abs(np.array(cl_kg_theory_clean)), '--', color='red', alpha=0.8, lw=2, label=r"Theory |$C_\ell^{\kappa g}$|")
            plt.title(r"Galaxy Auto & Cross Spectra")
        elif cl_gg_mean is not None:
            plt.title(r"Galaxy Auto Power Spectrum")

        plt.xlabel(r"$\ell$"), plt.ylabel(r"$C_\ell$"), plt.legend(), plt.grid(True, alpha=0.2)
        plt.xlim(10, 0.85 * ell_nyquist)  # Cut at 0.85*Nyquist (inference limit)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Make room for info box

    # Metadata Info Box
    cell_val = float(model.cell_shape[0]) if hasattr(model, "cell_shape") else model_config.get('cell_size', 0)
    box_x, box_y, box_z = float(model.box_shape[0]), float(model.box_shape[1]), float(model.box_shape[2])
    mesh_x, mesh_y, mesh_z = int(model.mesh_shape[0]), int(model.mesh_shape[1]), int(model.mesh_shape[2])

    info_str = f"Box: [{box_x:.0f}, {box_y:.0f}, {box_z:.0f}] Mpc/h | Mesh: ({mesh_x}, {mesh_y}, {mesh_z}) | Cell: {cell_val:.1f} Mpc/h"
    if cmb_enabled:
         info_str += f" | Field: {field_size:.1f}°"
    info_str += f" | N={n_realizations}"

    plt.figtext(0.5, 0.02, info_str, ha='center', fontsize=9,
                bbox={'facecolor': 'oldlace', 'alpha': 0.5, 'edgecolor': 'grey', 'boxstyle': 'round,pad=0.5'})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"cl_spectra{suffix}_{timestamp}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {outfile}")

    if show:
        plt.show()
    plt.close()

    return {"ell": ell, "cl_gg_mean": cl_gg_mean, "cl_kk_mean": cl_kk_box_mean}

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
