import os

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

from desi_cmb_fli.cmb_lensing import compute_signal_capture_ratio

# Adjust imports according to your package structure
from desi_cmb_fli.model import FieldLevelModel


def check_impact():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Redshifts to test (center of the box)
    redshifts = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Box sizes to test in Mpc/h
    # Note: We limit the max size to maintain decent resolution with fixed mesh
    box_sizes = np.linspace(256, 1536, 6)

    # Fixed mesh size for this test (higher than 64^3 for better fidelity)
    mesh_n = 128
    mesh_shape = (mesh_n, mesh_n, mesh_n)

    # Source redshift for CMB lensing
    z_source = 1100.0

    # Dummy Noise Power Spectrum for model initialization
    # (Required by __post_init__ but not used for the signal ratio calculation)
    dummy_ell = np.arange(1, 5000)
    dummy_nell_spec = np.ones_like(dummy_ell) * 1e-10
    dummy_nell_config = {"ell": dummy_ell, "N_ell": dummy_nell_spec}

    # Output directory
    if not os.path.exists("figures"):
        os.makedirs("figures")

    print("Running Signal Capture Check")
    print(f"Mesh Resolution: {mesh_n}^3")
    print("------------------------------------------------")

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    results = {}

    for z_obs in redshifts:
        a_obs = 1.0 / (1.0 + z_obs)

        # Get comoving distance for this redshift (using Planck15 as approx)
        cosmo_ref = jc.Planck15()
        chi_center = float(jc.background.radial_comoving_distance(cosmo_ref, jnp.atleast_1d(a_obs))[0])

        print(f"\nProcessing Redshift z={z_obs} (chi ~ {chi_center:.0f} Mpc/h)")

        ratios_for_z = []

        for box_size in box_sizes:
            # Calculate resolution
            res_mpc = box_size / mesh_n

            # 1. Initialize the FieldLevelModel
            # We use the model to generate the kappa map to ensure correct units/normalization
            model = FieldLevelModel(
                box_shape=np.array([box_size, box_size, box_size]),
                mesh_shape=np.array(mesh_shape),
                evolution="lpt",  # LPT is sufficient and faster for this check
                a_obs=a_obs,
                cmb_enabled=True,
                cmb_z_source=z_source,
                cmb_noise_nell=dummy_nell_config, # Dummy noise to satisfy init
                use_reduced_noise_for_closure_test=False
            )

            # 2. Generate a prediction (realization)
            # We use a fixed seed to compare similar structures across box sizes
            rng = jax.random.key(42)

            # We must condition on fiducial parameters to get the correct signal amplitude
            # model.loc_fid contains the fiducial values for all parameters that have them

            prediction = model.predict(
                rng=rng,
                samples=model.loc_fid,
                hide_det=False,
                hide_samp=False,
                frombase=True
            )

            kappa_map = prediction["kappa_pred"]

            # 3. Compute the Ratio
            # We construct the fiducial cosmology object directly
            from desi_cmb_fli.bricks import get_cosmology

            # Filter location params that are part of the 'cosmo' group
            cosmo_params = {k: model.loc_fid[k] for k in model.groups["cosmo"] if k in model.loc_fid}
            cosmo_fid = get_cosmology(**cosmo_params)

            ratio_2d = compute_signal_capture_ratio(
                cosmo_fid,
                kappa_map,
                model.cmb_field_size_deg,
                model.cmb_field_npix,
                z_source
            )

            # Metric: Mean ratio across the map
            mean_ratio = np.mean(ratio_2d)
            ratios_for_z.append(mean_ratio)

            print(f"  Box L={box_size:.0f} Mpc/h (Res: {res_mpc:.1f} Mpc) -> Capture: {mean_ratio*100:.3f}%")

        results[z_obs] = ratios_for_z

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for z_obs, ratios in results.items():
        plt.plot(box_sizes, np.array(ratios) * 100, 'o-', linewidth=2, label=f'z={z_obs}')

    plt.xlabel("Box Size (Mpc/h)")
    plt.ylabel("Mean Signal Capture Ratio (%)")
    plt.title(f"CMB Lensing Signal Capture vs Box Size (Mesh {mesh_n}$^3$)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()

    output_path = "figures/box_size_impact.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    check_impact()
