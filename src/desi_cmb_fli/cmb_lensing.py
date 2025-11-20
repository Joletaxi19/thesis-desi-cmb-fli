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
        plane = (plane - 1.0) * constant_factor * density_normalization

        nx = density_field.shape[0]
        pixel_coords = coords * r_planes[i] / dx + nx / 2.0
        im = map_coordinates(plane, pixel_coords, order=1, mode="wrap")

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

    # Use .shape directly (tuple) to avoid JAX tracing issues in jnp.arange
    mesh_shape = density_field.shape

    # Compute comoving distances for each slice
    dx = box_size[0] / mesh_shape[0]  # transverse cell size
    dz = box_size[2] / mesh_shape[2]  # radial cell size

    # Radial positions of slices (comoving distance)
    r_planes = (jnp.arange(mesh_shape[2]) + 0.5) * dz

    # Convert to scale factors (assuming box aligned with line of sight)
    a_planes = jc.background.a_of_chi(cosmo, r_planes)

    # Create angular coordinate grid (CENTERED on the box)
    # map_coordinates expects coords[0]=rows (y), coords[1]=cols (x)
    # Use indexing='ij' for direct (row, col) ordering
    rowgrid, colgrid = np.meshgrid(
        np.linspace(-field_size_deg/2, field_size_deg/2, field_npix, endpoint=False),
        np.linspace(-field_size_deg/2, field_size_deg/2, field_npix, endpoint=False),
        indexing='ij'
    )
    # With indexing='ij': rowgrid[i,j] varies with i, colgrid[i,j] varies with j
    # Convert to radians AFTER stacking to avoid multiplying axis dimension
    coords = jnp.array(np.stack([rowgrid, colgrid], axis=0)) * u.deg.to(u.rad)

    # Compute convergence
    z_source_array = jnp.atleast_1d(z_source)
    is_scalar = jnp.ndim(z_source) == 0

    kappa = convergence_Born(
        cosmo,
        density_field,
        r_planes,
        a_planes,
        dx,
        dz,
        coords,
        z_source_array
    )

    # Squeeze if z_source was scalar
    if is_scalar:
        kappa = kappa.squeeze(0)

    return kappa
