"""
CMB Lensing Module - Integrated with existing field-level inference

This module provides CMB lensing observables that can be computed from
the matter density field inferred from galaxy clustering.
"""

import astropy.units as u
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates


def convergence_Born(cosmo,
                     density_field,
                     r_planes,
                     a_planes,
                     dx,
                     dz,
                     coords,
                     z_source):
    """
    Compute the Born convergence from a 3D density field.

    This is adapted to work with the existing density fields from FieldLevelModel.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object
    density_field : array [Nx, Ny, Nz]
        3D density contrast field (1+delta)
    r_planes : array [Nz]
        Comoving distances for each z-slice
    a_planes : array [Nz]
        Scale factors for each z-slice
    dx : float
        Transverse cell size (Mpc/h)
    dz : float
        Radial cell size (Mpc/h)
    coords : array [2, Npix_x, Npix_y]
        Angular coordinates in radians
    z_source : array
        Source redshifts

    Returns
    -------
    convergence : array [Npix_x, Npix_y, Nz_source]
        Convergence maps for each source redshift
    """
    # Compute constant prefactor: 3/2 * Omega_m * (H0/c)^2
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2

    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    n_planes = len(r_planes)

    def scan_fn(carry, i):
        density_field, a_planes, r_planes = carry

        # Extract the i-th plane and normalize
        plane = density_field[:, :, i]
        density_normalization = dz * r_planes[i] / a_planes[i]
        plane = (plane - plane.mean()) * constant_factor * density_normalization

        # Interpolate at the requested angular coordinates
        # coords * r gives physical positions, divide by dx to get pixel coordinates
        im = map_coordinates(plane,
                           coords * r_planes[i] / dx - 0.5,
                           order=1, mode="wrap")

        # Weight by lensing kernel: (1 - r/r_s)
        kernel = jnp.clip(1. - (r_planes[i] / r_s), 0, 1000).reshape([-1, 1, 1])

        return carry, im * kernel

    _, convergence_planes = jax.lax.scan(
        scan_fn,
        (density_field, a_planes, r_planes),
        jnp.arange(n_planes)
    )

    return convergence_planes.sum(axis=0)


def density_field_to_convergence(density_field,
                                 box_size,
                                 cosmo,
                                 field_size_deg,
                                 field_npix,
                                 z_source=1.0):
    """
    Convert a 3D density field from FieldLevelModel to a CMB convergence map.

    This function bridges your existing galaxy clustering inference with CMB lensing.

    Parameters
    ----------
    density_field : array [Nx, Ny, Nz]
        3D density field (1+delta) from FieldLevelModel
    box_size : array [3]
        Physical box size in Mpc/h
    cosmo : jax_cosmo.Cosmology
        Cosmology object
    field_size_deg : float
        Angular size of the field in degrees
    field_npix : int
        Number of pixels for the convergence map
    z_source : float or array
        Source redshift(s)

    Returns
    -------
    kappa : array [field_npix, field_npix] or [field_npix, field_npix, Nz]
        Convergence map(s)
    """
    import numpy as np

    mesh_shape = jnp.array(density_field.shape)

    # Compute comoving distances for each slice
    dx = box_size[0] / mesh_shape[0]  # transverse cell size
    dz = box_size[2] / mesh_shape[2]  # radial cell size

    # Radial positions of slices (comoving distance)
    r_planes = (jnp.arange(mesh_shape[2]) + 0.5) * dz

    # Convert to scale factors (assuming box aligned with line of sight)
    a_planes = jc.background.a_of_chi(cosmo, r_planes)

    # Create angular coordinate grid
    xgrid, ygrid = np.meshgrid(
        np.linspace(0, field_size_deg, field_npix, endpoint=False),
        np.linspace(0, field_size_deg, field_npix, endpoint=False)
    )
    coords = jnp.array(np.stack([xgrid, ygrid], axis=0) * u.deg.to(u.rad))

    # Compute convergence
    z_source = jnp.atleast_1d(z_source)
    kappa = convergence_Born(
        cosmo,
        density_field,
        r_planes,
        a_planes,
        dx,
        dz,
        coords,
        z_source
    )

    return kappa


def pixel_window_function(ell, pixel_size_arcmin):
    """
    Calculate the pixel window function for a given multipole and pixel size.

    Parameters
    ----------
    ell : array
        Multipole values
    pixel_size_arcmin : float
        Pixel size in arcminutes

    Returns
    -------
    W_ell : array
        Pixel window function
    """
    import numpy as np

    # Convert pixel size from arcminutes to radians
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))

    # Sinc^2 window function
    W_ell = (jnp.sinc(ell * pixel_size_rad / (2 * np.pi)))**2

    return W_ell


def theoretical_cl(cosmo, nz_shear, ell, pixel_scale_arcmin, sigma_e=0.3):
    """
    Compute theoretical angular power spectrum for weak lensing.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object
    nz_shear : jax_cosmo.redshift.NzRedshift
        Redshift distribution of sources
    ell : array
        Multipole values
    pixel_scale_arcmin : float
        Pixel scale in arcminutes
    sigma_e : float
        Intrinsic ellipticity dispersion

    Returns
    -------
    cl_theory : array
        Theoretical C_ell including pixel window
    cl_noise : array
        Noise C_ell
    """
    # Compute angular power spectrum
    tracer = jc.probes.WeakLensing([nz_shear], sigma_e=sigma_e)
    cl_theory = jc.angular_cl.angular_cl(
        cosmo, ell, [tracer],
        nonlinear_fn=jc.power.linear
    )

    # Apply pixel window function
    cl_theory = cl_theory * pixel_window_function(ell, pixel_scale_arcmin)

    # Compute noise
    cl_noise = jc.angular_cl.noise_cl(ell, [tracer])

    return cl_theory, cl_noise


# Integration with FieldLevelModel
def add_cmb_lensing_observable(field_level_model,
                               field_size_deg=5.0,
                               field_npix=64,
                               z_source=1.0):
    """
    Extend FieldLevelModel predictions to include CMB lensing convergence.

    This should be called after evolving the model to get density field.

    Parameters
    ----------
    field_level_model : FieldLevelModel
        Your existing field-level model
    field_size_deg : float
        Angular field size in degrees
    field_npix : int
        Number of pixels for convergence map
    z_source : float or array
        Source redshift(s)

    Returns
    -------
    predict_with_lensing : function
        Modified prediction function that includes convergence maps
    """

    def predict_with_lensing(rng, samples=None, **kwargs):
        # Get standard predictions (galaxy field)
        predictions = field_level_model.predict(rng, samples, **kwargs)

        # If we have a 3D density field, compute convergence
        if 'gxy_mesh' in predictions:
            # Extract cosmology from predictions
            cosmo_params = {k: predictions[k] for k in ['Omega_m', 'sigma8']
                          if k in predictions}

            from desi_cmb_fli.bricks import get_cosmology
            cosmo = get_cosmology(**cosmo_params)

            # Convert galaxy mesh to convergence
            # Note: gxy_mesh is 1+delta_gxy, we need matter density
            # Assuming bias b1 is available
            b1 = predictions.get('b1', 1.0)
            delta_matter = (predictions['gxy_mesh'] - 1.0) / b1
            density_field = 1.0 + delta_matter

            # Compute convergence
            kappa = density_field_to_convergence(
                density_field,
                field_level_model.box_shape,
                cosmo,
                field_size_deg,
                field_npix,
                z_source
            )

            predictions['kappa'] = kappa

        return predictions

    return predict_with_lensing
