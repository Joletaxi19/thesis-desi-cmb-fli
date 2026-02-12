import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa


def check_derivative():
    # Fixed parameters
    sigma8_fix = 0.8
    ell = jnp.logspace(2, 3.5, 50)

    # Many more Omega_m values for smooth gradient
    om_values = jnp.linspace(0.25, 0.35, 25)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=om_values.min(), vmax=om_values.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    print(f"\n{'='*60}")
    print(f"Testing Omega_m variation (sigma_8={sigma8_fix}, h=0.6766 fixed)")
    print(f"{'='*60}")

    for val in om_values:
        # Create cosmology
        cosmo = get_cosmology(Omega_m=val, sigma8=sigma8_fix)

        # Compute theoretical Cl
        cl = compute_theoretical_cl_kappa(
            cosmo, ell,
            chi_min=1.0, chi_max=14000.0,
            z_source=1100.0
        )

        col = cmap(norm(val))
        ax.loglog(ell, cl, '-', color=col, linewidth=2.0, alpha=0.85)

    print(f"{'='*60}\n")

    ax.set_xlabel('$\\ell$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$C_\\ell^{\\kappa\\kappa}$', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('$\\Omega_m$', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(f'Effect of $\\Omega_m$ on CMB lensing ($\\sigma_8 = {sigma8_fix}$ fixed)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig("figures/positive_correlation_diagnostic/omega_m_effect.png", dpi=300, bbox_inches='tight')
    print("Figure saved: figures/positive_correlation_diagnostic/omega_m_effect.png")

if __name__ == "__main__":
    check_derivative()
