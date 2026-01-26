#!/usr/bin/env python
"""
Plot comparison of CMB lensing noise spectra: ACT DR6 vs Planck PR4.

This script loads both noise power spectra and creates a comparison plot
showing the improvement provided by ACT DR6 over Planck PR4.

Usage:
    python scripts/plot_cmb_noise_comparison.py [--output OUTPUT_DIR]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_planck_nlkk(filepath="/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat"):
    """Load Planck PR4 N_ell^kappa_kappa."""
    data = np.loadtxt(filepath)
    ell = np.arange(len(data))
    nlkk = data
    return ell, nlkk


def load_act_nlkk(filepath="data/N_L_kk_act_dr6_lensing_v1_baseline.txt"):
    """Load ACT DR6 N_ell^kappa_kappa."""
    data = np.loadtxt(filepath)
    ell = data[:, 0].astype(int)
    nlkk = data[:, 1]
    return ell, nlkk


def plot_comparison(output_dir="figures"):
    """Create comparison plot of ACT vs Planck noise spectra."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading noise spectra...")
    planck_ell, planck_nlkk = load_planck_nlkk()
    act_ell, act_nlkk = load_act_nlkk()

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ====== Subplot 1: N_ell comparison ======
    ax = axes[0]

    # Plot Planck
    mask_planck = (planck_ell >= 2) & (planck_ell <= 3000) & (planck_nlkk > 0)
    ax.loglog(planck_ell[mask_planck], planck_nlkk[mask_planck],
              label='Planck PR4', lw=2.5, alpha=0.8, color='C0')

    # Plot ACT
    mask_act = (act_ell >= 2) & (act_ell <= 3000) & (act_nlkk > 0)
    ax.loglog(act_ell[mask_act], act_nlkk[mask_act],
              label='ACT DR6 baseline', lw=2.5, alpha=0.8, color='C1')

    ax.set_ylabel(r'$N_\ell^{\kappa\kappa}$', fontsize=13)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11)
    ax.grid(True, which='both', alpha=0.3, ls='-', lw=0.5)
    ax.set_title('CMB Lensing Reconstruction Noise Comparison',
                 fontsize=14, fontweight='bold', pad=10)

    # ====== Subplot 2: Ratio ACT/Planck ======
    ax = axes[1]

    # Compute ratio on common ell range
    ell_common = np.arange(2, min(2049, len(planck_nlkk), len(act_nlkk)))
    planck_vals = planck_nlkk[ell_common]
    act_vals = act_nlkk[ell_common]
    mask_valid = (planck_vals > 0) & (act_vals > 0)

    ell_plot = ell_common[mask_valid]
    ratio = act_vals[mask_valid] / planck_vals[mask_valid]

    ax.semilogx(ell_plot, ratio, lw=2.5, color='C2', label=f'ACT/Planck (mean={ratio.mean():.2f})')
    ax.axhline(1, color='black', ls='--', lw=1.5, alpha=0.5, label='Equal noise')
    ax.axhline(ratio.mean(), color='C2', ls=':', lw=2, alpha=0.7)

    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'$N_\ell^{\rm ACT} / N_\ell^{\rm Planck}$', fontsize=13)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, which='both', alpha=0.3, ls='-', lw=0.5)
    ax.set_ylim([0, 1.2])

    # Add shaded region showing ACT is better
    ax.fill_between(ell_plot, 0, ratio, alpha=0.2, color='C2',
                     label='ACT improvement region')

    plt.tight_layout()

    # Save figure
    outfile = output_dir / "cmb_noise_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {outfile}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Ell range: {ell_plot.min()} - {ell_plot.max()}")
    print(f"Mean ratio (ACT/Planck): {ratio.mean():.3f}")
    print(f"ACT has {1/ratio.mean():.2f}× lower noise than Planck")

    print("\nAt specific multipoles:")
    for ell_test in [100, 500, 1000, 2000]:
        if ell_test < len(planck_nlkk) and ell_test < len(act_nlkk):
            if planck_nlkk[ell_test] > 0 and act_nlkk[ell_test] > 0:
                r = act_nlkk[ell_test] / planck_nlkk[ell_test]
                print(f"  ℓ={ell_test:4d}: ACT/Planck = {r:.3f}")
    print("="*60)

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot ACT DR6 vs Planck PR4 CMB lensing noise comparison"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="figures",
        help="Output directory for figure (default: figures/)"
    )
    args = parser.parse_args()

    plot_comparison(output_dir=args.output)


if __name__ == "__main__":
    main()
