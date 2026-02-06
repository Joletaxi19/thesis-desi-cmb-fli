import os

import jax.numpy as jnp
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
    if 'field_npix' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['field_npix'] = 256

    # Instantiate model to get grid geometry and noise
    model, _ = get_model_from_config(cfg)

    print("Computing gradients and Fisher info (this may take a moment)...")

    # 2. Setup Variables
    ell_2d = model.ell_grid
    noise_2d = model.cmb_noise_power_k
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

    # Filter valid modes (remove DC mode l=0 and very high l where noise dominates)
    mask = (ell_2d > 20) & (ell_2d < 3000) & \
           jnp.isfinite(dCl_dOm_2d) & jnp.isfinite(Cl_2d) & jnp.isfinite(noise_2d)

    ell_flat = ell_2d[mask]
    dCl_flat = dCl_dOm_2d[mask]
    Cl_flat = Cl_2d[mask]
    Nl_flat = noise_2d[mask]

    fisher_per_pixel = 0.5 * (dCl_flat / (Cl_flat + Nl_flat))**2

    # Total Information (Sum over all pixels)
    total_fisher = jnp.sum(fisher_per_pixel)
    predicted_sigma = 1.0 / jnp.sqrt(total_fisher)

    print("-" * 50)
    print(f"RESULTS FOR PATCH SIZE: {model.cmb_field_size_deg:.2f} deg")
    print("-" * 50)
    print(f"Total Fisher Info (Omega_m): {total_fisher:.2f}")
    print(f"Predicted 1-sigma error on Omega_m: {predicted_sigma:.5f}")
    print("-" * 50)

    # 5. Binning for Plotting (1D representation)
    # We aggregate the information from the 2D pixels into 1D ell-bins
    n_bins = 50
    l_min, l_max = 20, 3000
    l_edges = np.linspace(l_min, l_max, n_bins + 1)
    l_centers = 0.5 * (l_edges[1:] + l_edges[:-1])

    # Sum Fisher info in each bin
    fisher_1d, _ = np.histogram(ell_flat, bins=l_edges, weights=fisher_per_pixel)

    # To calculate the sign of the derivative for the plot color (red/green)
    # We need a weighted average of the sign in each bin
    sign_weights = np.sign(dCl_flat) * fisher_per_pixel
    sign_1d, _ = np.histogram(ell_flat, bins=l_edges, weights=sign_weights)
    # If the accumulated weighted sign is negative, we paint red

    # 6. Plotting
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Plot positive gradients (Green)
    mask_pos = sign_1d >= 0
    plt.bar(l_centers[mask_pos], fisher_1d[mask_pos], width=(l_max-l_min)/n_bins,
            color='#2ca02c', alpha=0.7, label=r'$\partial C_\ell / \partial\Omega_m > 0$')

    # Plot negative gradients (Red)
    mask_neg = sign_1d < 0
    plt.bar(l_centers[mask_neg], fisher_1d[mask_neg], width=(l_max-l_min)/n_bins,
            color='#d62728', alpha=0.7, label=r'$\partial C_\ell / \partial\Omega_m < 0$')

    plt.xlabel(r"Multipole $\ell$", fontsize=14)
    plt.ylabel(r"Fisher Info contribution $\Delta F(\ell)$", fontsize=14)
    plt.title(rf"Fisher Information for $\Omega_m$ (Patch: {model.cmb_field_size_deg}$^\circ$)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(l_min, l_max)

    # Stats box
    textstr = '\n'.join((
        rf'$\sigma(\Omega_m) \approx {predicted_sigma:.4f}$',
        rf'$N_{{pix}} = {model.cmb_field_npix}^2$'))
    props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
    plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    filename = f"figures/positive_correlation_diagnostic/fisher{model.cmb_field_npix}.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Figure saved: {filename}")

if __name__ == "__main__":
    simple_fisher()
