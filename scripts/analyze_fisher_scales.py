import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jax import jacfwd

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa
from desi_cmb_fli.model import get_model_from_config


def simple_fisher(config_path="configs/inference/config.yaml"):
    # 1. Load model to get ell grid and noise
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if 'cmb_lensing' not in cfg:
        cfg['cmb_lensing'] = {}

    cfg['cmb_lensing']['enabled'] = True
    cfg['cmb_lensing']['full_los_correction'] = False  # We'll compute full Cl manually

    if 'cmb_noise_nell' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['cmb_noise_nell'] = "data/N_L_kk_act_dr6_lensing_v1_baseline.txt"
    if 'z_source' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['z_source'] = 1100.0
    if 'field_npix' not in cfg['cmb_lensing']:
        cfg['cmb_lensing']['field_npix'] = 256

    model, _ = get_model_from_config(cfg)

    print("Computing total C_l (z=0 to z_CMB) and gradients...")

    # 2. Compute total Cl and gradients (full line of sight)
    ell_2d = model.ell_grid
    noise_2d = model.cmb_noise_power_k
    loc_fid = model.loc_fid

    # Wrapper function for differentiation
    def get_cl_total(theta):
        Om, s8 = theta
        cosmo = get_cosmology(Omega_m=Om, sigma8=s8)
        # full line of sight: chi_min = 1.0 (near z=0)
        return compute_theoretical_cl_kappa(
            cosmo, ell_2d,
            chi_min=1.0,  # Start from z≈0
            chi_max=14000.0,  # To z_CMB
            z_source=1100.0
        )

    # Fiducial point
    theta_fid = jnp.array([loc_fid["Omega_m"], loc_fid["sigma8"]])

    # Compute Cl at fiducial
    Cl_2d = get_cl_total(theta_fid)

    # Compute Jacobian (gradients)
    jac_fn = jacfwd(get_cl_total)
    grads = jac_fn(theta_fid)
    dCl_dOm_2d = grads[..., 0]

    # 3. Extract data (flatten 2D grids)
    ell = ell_2d.flatten()
    dCl = dCl_dOm_2d.flatten()
    Cl = Cl_2d.flatten()
    Nl = noise_2d.flatten()

    # 4. Filter and Sort (for plotting)
    mask = (ell > 20) & (ell < 3000) & np.isfinite(dCl) & np.isfinite(Cl) & np.isfinite(Nl)
    ell, dCl, Cl, Nl = ell[mask], dCl[mask], Cl[mask], Nl[mask]

    idx = np.argsort(ell)
    ell, dCl, Cl, Nl = ell[idx], dCl[idx], Cl[idx], Nl[idx]

    # 5. Fisher Information calculation (per mode l)
    Var = Cl + Nl
    Fisher_per_mode = (dCl / Var)**2

    # Total info per multipole l (sum over 2l+1 modes m)
    w_smooth = np.convolve(Fisher_per_mode * (2*ell+1), np.ones(50)/50, mode='same')

    plt.figure(figsize=(9, 5))

    # Red Zone vs Green
    sign = np.sign(dCl)
    plt.fill_between(ell, 0, w_smooth, where=(sign < 0), color='#d62728', alpha=0.7,
                     label=r'$\partial C_\ell / \partial\Omega_m < 0$')
    plt.fill_between(ell, 0, w_smooth, where=(sign > 0), color='#2ca02c', alpha=0.7,
                     label=r'$\partial C_\ell / \partial\Omega_m > 0$')

    plt.xlabel(r"$\ell$", fontsize=14)
    plt.ylabel(r"Fisher Information $(2\ell+1) \left[\frac{\partial C_\ell^{\kappa\kappa}}{\partial\Omega_m}\right]^2 / \left(C_\ell^{\kappa\kappa,\mathrm{total}} + N_\ell^{\kappa\kappa,\mathrm{ACT}}\right)^2$", fontsize=13)
    plt.title(r"Fisher information for $\Omega_m$",
              fontsize=14, pad=12)
    plt.legend(fontsize=12, framealpha=0.95, loc='upper right')
    plt.grid(alpha=0.25, linestyle='-', linewidth=0.5)
    plt.xlim(ell.min(), ell.max())
    plt.yscale('log')
    plt.ylim(bottom=max(w_smooth[w_smooth > 0].min() * 0.5, 1e-2))
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig("figures/fisher.png", dpi=300, bbox_inches='tight')
    print("✓ Figure saved: figures/fisher.png")

if __name__ == "__main__":
    simple_fisher()
