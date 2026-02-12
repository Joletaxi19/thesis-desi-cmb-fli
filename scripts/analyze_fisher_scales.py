import os

import jax.numpy as jnp
import jax_cosmo
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jax import jacfwd

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa
from desi_cmb_fli.model import get_model_from_config


def simple_fisher(config_path="configs/inference/config.yaml"):
    print(f"Loading config from {config_path}...")

    # 1. Load model configuration
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Force CMB lensing enabled and setup for theoretical calculation
    if 'cmb_lensing' not in cfg:
        cfg['cmb_lensing'] = {}

    cfg['cmb_lensing']['enabled'] = True
    cfg['cmb_lensing']['full_los_correction'] = False  # We compute full Cl manually here

    # Defaults if missing
    if 'cmb_noise_nell' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['cmb_noise_nell'] = "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"
    if 'z_source' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['z_source'] = 1100.0

    # Also enable galaxies for comparison (constructor reads model.galaxies_enabled)
    cfg['model'] = cfg.get('model', {})
    cfg['model']['galaxies_enabled'] = True

    # Instantiate model to get grid geometry and noise
    model, _ = get_model_from_config(cfg)

    print("Computing gradients and Fisher info (this may take a moment)...")

    # 2. Setup Variables
    ell_2d = model.ell_grid
    noise_2d = model.nell_grid
    loc_fid = model.loc_fid

    # Wrapper function for differentiation (Total C_l from z=0 to z_CMB)
    def get_cl_total(theta):
        Om, s8 = theta
        # Reconstruct cosmology from params
        cosmo = get_cosmology(Omega_m=Om, sigma8=s8)

        # Integrate from chi_min=1.0 (approx z=0) to chi_max=14000 (z_CMB)
        # This captures the total lensing signal (Box + High-Z)
        return compute_theoretical_cl_kappa(
            cosmo, ell_2d,
            chi_min=1.0,
            chi_max=14000.0,
            z_source=cfg['cmb_lensing']['z_source']
        )

    # Fiducial point
    theta_fid = jnp.array([loc_fid["Omega_m"], loc_fid["sigma8"]])

    # 3. Compute Jacobian (Automatic Differentiation)
    # This computes dCl / dTheta exactly
    Cl_2d = get_cl_total(theta_fid)
    jac_fn = jacfwd(get_cl_total)
    grads = jac_fn(theta_fid)

    dCl_dOm_2d = grads[..., 0] # Derivative w.r.t Omega_m
    # dCl_ds8_2d = grads[..., 1] # Derivative w.r.t sigma8 (unused for this plot)

    # 4. Fisher Information Calculation

    # Use Nyquist limit
    ell_nyquist = model.ell_nyquist
    print(f"Nyquist limit from model: {ell_nyquist:.1f}")
    print(f"Max ell in grid: {jnp.max(ell_2d):.1f}")

    # Filter valid modes (remove DC mode and modes beyond 0.85*Nyquist, matching likelihood mask)
    mask = (ell_2d > 20) & (ell_2d <= 0.85 * ell_nyquist) & \
           jnp.isfinite(dCl_dOm_2d) & jnp.isfinite(Cl_2d) & jnp.isfinite(noise_2d)

    ell_flat = ell_2d[mask]
    dCl_flat = dCl_dOm_2d[mask]
    Cl_flat = Cl_2d[mask]
    Nl_flat = noise_2d[mask]

    fisher_per_pixel = 0.5 * (dCl_flat / (Cl_flat + Nl_flat))**2

    # Total Information (Sum over all pixels)
    total_fisher_cmb = jnp.sum(fisher_per_pixel)
    predicted_sigma_cmb = 1.0 / jnp.sqrt(total_fisher_cmb)

    print("-" * 50)
    print(f"CMB LENSING RESULTS (Patch: {model.cmb_field_size_deg:.2f} deg)")
    print("-" * 50)
    print(f"Total Fisher Info (Omega_m): {total_fisher_cmb:.2f}")
    print(f"Predicted 1-sigma error on Omega_m: {predicted_sigma_cmb:.5f}")
    print("-" * 50)

    # ===== GALAXY FISHER CALCULATION (THEORETICAL) =====
    print("\nComputing galaxy Fisher information (theoretical)...")

    b1 = loc_fid.get("b1", 2.0)  # Lagrangian bias
    bE = 1.0 + b1  # Eulerian bias: b_E = 1 + b_L

    V_survey = float(np.prod(model.box_shape))  # (Mpc/h)³
    nbar = model.gxy_density  # galaxies per (Mpc/h)^3

    box_lengths = np.asarray(model.box_shape, dtype=float)
    mesh_sizes = np.asarray(model.mesh_shape, dtype=int)
    k_fund_axes = 2.0 * np.pi / box_lengths
    k_nyquist_axes = np.pi * mesh_sizes / box_lengths

    # Exact anisotropic mode lattice from FFT frequencies:
    # k_i = 2π * fftfreq(N_i, d=L_i/N_i)
    kx = 2.0 * np.pi * np.fft.fftfreq(mesh_sizes[0], d=box_lengths[0] / mesh_sizes[0])
    ky = 2.0 * np.pi * np.fft.fftfreq(mesh_sizes[1], d=box_lengths[1] / mesh_sizes[1])
    kz = 2.0 * np.pi * np.fft.fftfreq(mesh_sizes[2], d=box_lengths[2] / mesh_sizes[2])
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Keep all physical non-DC modes present on the discrete lattice
    k_modes = jnp.asarray(k_mag.ravel())
    k_mode_mask = k_modes > 0.0
    k_modes = k_modes[k_mode_mask]

    def get_Pgg_all_k(theta):
        """Vectorized P_gg(k) = b_E² * P_mm(k) from jax_cosmo."""
        Om, s8 = theta
        cosmo = get_cosmology(Omega_m=Om, sigma8=s8)
        # Vectorized over all k_modes at once
        Pmm = jnp.squeeze(jax_cosmo.power.linear_matter_power(cosmo, k_modes, a=model.a_obs))
        return (bE**2) * Pmm

    # Compute P_gg(k) and its derivative via JAX autodiff
    Pgg_k = get_Pgg_all_k(theta_fid)
    dPgg_dtheta = jacfwd(get_Pgg_all_k)(theta_fid)
    dPgg_dOm = dPgg_dtheta[..., 0]  # Derivative w.r.t Omega_m

    # FKP effective-volume weight per mode: [n̄P/(1 + n̄P)]²
    fkp_weight_sq = (nbar * Pgg_k / (1.0 + nbar * Pgg_k))**2

    # Fisher integrand: (1/2) * [∂lnP/∂Ω_m]² * [n̄P/(1+n̄P)]²
    dlnP_dOm = dPgg_dOm / Pgg_k
    fisher_integrand = 0.5 * fkp_weight_sq * dlnP_dOm**2

    # Discrete exact finite-volume Fisher sum:
    # F = (1/2) * Σ_k [n̄P/(1+n̄P)]² [∂lnP/∂Ω_m]²
    total_fisher_gxy = jnp.sum(fisher_integrand)

    predicted_sigma_gxy = 1.0 / jnp.sqrt(total_fisher_gxy)

    z_obs = 1.0/model.a_obs - 1.0

    print("-" * 50)
    print(f"GALAXY RESULTS (Box shape: {tuple(model.box_shape)} Mpc/h)")
    print("-" * 50)
    print(f"Galaxy density n̄: {nbar:.4e} (Mpc/h)^-3")
    print(f"Volume: {V_survey:.1f} (Mpc/h)^3")
    print(f"Redshift z_obs: {z_obs:.4f}")
    print(f"Lagrangian bias b1: {b1:.2f}")
    print(f"Eulerian bias bE: {bE:.2f}")
    print(f"k_fund per axis: {k_fund_axes} h/Mpc")
    print(f"k_nyquist per axis: {k_nyquist_axes} h/Mpc")
    print(f"Discrete k-mode count (non-DC): {k_modes.size}")
    print(f"k range used: [{jnp.min(k_modes):.4f}, {jnp.max(k_modes):.4f}] h/Mpc")
    print(f"Total Fisher Info (Omega_m): {total_fisher_gxy:.2f}")
    print(f"Predicted 1-sigma error on Omega_m: {predicted_sigma_gxy:.5f}")
    print("-" * 50)

    # 5. Binning for Plotting (1D representation)
    n_bins = 20
    l_min = 20
    l_max = float(0.85 * ell_nyquist)  # Match the mask used in likelihood
    l_edges = np.linspace(l_min, l_max, n_bins + 1)
    l_centers = 0.5 * (l_edges[1:] + l_edges[:-1])

    print(f"Using {n_bins} bins from ell={l_min} to ell={l_max:.1f}")
    print(f"Number of valid modes: {jnp.sum(mask)}")

    # Sum Fisher info in each bin
    fisher_1d, _ = np.histogram(ell_flat, bins=l_edges, weights=fisher_per_pixel)

    # To calculate the sign of the derivative for the plot color (red/green)
    sign_weights = np.sign(dCl_flat) * fisher_per_pixel
    sign_1d, _ = np.histogram(ell_flat, bins=l_edges, weights=sign_weights)

    # 6. Plotting
    os.makedirs("figures/positive_correlation_diagnostic", exist_ok=True)

    # Single figure for clarity
    plt.figure(figsize=(10, 7))

    width = (l_max - l_min) / n_bins

    # Plot positive gradients (Green)
    mask_pos = sign_1d >= 0
    plt.bar(l_centers[mask_pos], fisher_1d[mask_pos], width=width,
            color='#2ca02c', alpha=0.7, edgecolor='none', label=r'$\partial C_\ell / \partial\Omega_m > 0$')

    # Plot negative gradients (Red)
    mask_neg = sign_1d < 0
    plt.bar(l_centers[mask_neg], fisher_1d[mask_neg], width=width,
            color='#d62728', alpha=0.7, edgecolor='none', label=r'$\partial C_\ell / \partial\Omega_m < 0$')

    plt.xlabel(r"Multipole $\ell$", fontsize=14)
    plt.ylabel(r"$\Delta F(\ell) = \frac{1}{2} \, (\frac{\partial C_\ell}{\partial \Omega_m})^2 \, (C_\ell^{\kappa\kappa} + N_\ell)^{-2}$", fontsize=12)
    plt.title(rf"CMB Lensing Information (Patch: {model.cmb_field_size_deg:.1f}$^\circ$)", fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.2)
    plt.xlim(l_min, l_max)

    # Comparison Info Box
    # Moved to the right, below the legend
    ratio = total_fisher_gxy/total_fisher_cmb

    # Mathematical formulas for annotation
    formula_cmb = r"$\mathbf{CMB:}\ F_{CMB} = \sum_\ell \Delta F(\ell)$"
    formula_gxy = r"$\mathbf{Galaxy:}\ F_{gxy} = \frac{1}{2}\sum_{\mathbf{k}\neq 0}\left[\frac{nP_{gg}(k)}{1+nP_{gg}(k)}\right]^2\left[\frac{\partial \ln P_{gg}(k)}{\partial\Omega_m}\right]^2$"

    textstr = '\n'.join((
        r'$\bf{Fisher\ Information}$',
        '',
        formula_cmb,
        rf'$\quad F_{{CMB}} = {total_fisher_cmb:.1f}$',
        rf'$\quad \sigma_{{CMB}} = {predicted_sigma_cmb:.4f}$',
        '',
        formula_gxy,
        rf'$\quad F_{{gxy}} \approx {total_fisher_gxy:.2e}$',
        rf'$\quad \sigma_{{gxy}} = {predicted_sigma_gxy:.4f}$',
        '',
        r'$\bf{Comparison}$',
        rf'Ratio $F_{{gxy}}/F_{{CMB}} \approx {ratio:.0f}\times$'
        ))

    props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'gray', 'linewidth': 1.0}

    plt.gca().text(0.97, 0.72, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    filename = f"figures/positive_correlation_diagnostic/fisher{model.cmb_field_npix}.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Figure saved: {filename}")

if __name__ == "__main__":
    simple_fisher()
