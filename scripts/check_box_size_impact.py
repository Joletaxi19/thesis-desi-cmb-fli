
import os

import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

from desi_cmb_fli.cmb_lensing import compute_lensing_efficiency


def check_impact():
    # Setup cosmology
    cosmo = jc.Planck15()

    # Parameters
    z_source = 1100.0
    a_obs = 0.333 # z=2
    chi_center = float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(a_obs))[0])

    mesh_sizes = [32, 64, 128, 256]
    resolutions = [4.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

    print(f"Testing Box Size Impact at z=2 (chi={chi_center:.1f} Mpc/h)")

    # Create figures directory if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.figure(figsize=(12, 8))

    for mesh_size in mesh_sizes:
        print(f"\n--- Mesh Size: {mesh_size}^3 ---")
        print(f"{'Resolution':<12} {'Box Size':<12} {'Coverage':<10}")
        print("-" * 40)

        results = []

        for res in resolutions:
            box_size_z = res * mesh_size
            z_grid, kernel, fraction, mask = compute_lensing_efficiency(cosmo, z_source, chi_center, box_size_z)

            print(f"{res:<12.1f} {box_size_z:<12.1f} {fraction*100:.2f}%")
            results.append((res, box_size_z, fraction))

        # Plot comparison
        res_vals = [r[0] for r in results]
        frac_vals = [r[2]*100 for r in results]

        # Label with mesh size and box size range
        min_box = results[0][1]
        max_box = results[-1][1]
        label = f"Mesh {mesh_size}$^3$ (Box: {min_box:.0f}-{max_box:.0f} Mpc/h)"

        plt.plot(res_vals, frac_vals, 'o-', linewidth=2, label=label)

    plt.xlabel("Resolution (Mpc/h)")
    plt.ylabel("Kernel Coverage (%)")
    plt.title("Lensing Kernel Coverage vs Resolution for different Mesh Sizes (z=2)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = "figures/box_size_impact.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    check_impact()
