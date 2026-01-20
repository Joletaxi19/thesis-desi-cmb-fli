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
    convergence : array [Nz_source, Npix_x, Npix_y]
        Convergence maps for each source redshift.
    """
    # Compute constant prefactor: 3/2 * Omega_m * (H0/c)^2
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2

    # Compute comoving distance of source(s)
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    n_planes = len(r_planes)

    def scan_fn(carry, i):
        density_field, a_planes, r_planes = carry

        # Extract the i-th plane and normalize
        plane = density_field[:, :, i]
        density_normalization = dz * r_planes[i] / a_planes[i]
        plane = (plane - 1.0) * constant_factor * density_normalization

        nx = density_field.shape[0]
        # Gnomonic projection: x_phys = r * tan(theta)
        # coords are in radians
        # pixel_coords = (x_phys / dx) + nx/2
        pixel_coords = jnp.tan(coords) * r_planes[i] / dx + nx / 2.0
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
                                 z_source=1.0,
                                 box_center_chi=None):
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
    box_center_chi : float, optional
        Comoving distance to the center of the box (Mpc/h).
        If None, the box is assumed to start at chi=0.

    Returns
    -------
    kappa : array [field_npix, field_npix] or [Nz_source, field_npix, field_npix]
        Convergence map(s). If z_source is a scalar, returns [Npix, Npix].
        If z_source is an array, returns [Nz_source, Npix, Npix].
    """
    import numpy as np

    # Use .shape directly (tuple) to avoid JAX tracing issues in jnp.arange
    mesh_shape = density_field.shape

    # Compute comoving distances for each slice
    dx = box_size[0] / mesh_shape[0]  # transverse cell size
    dz = box_size[2] / mesh_shape[2]  # radial cell size

    # Radial positions of slices (comoving distance)
    if box_center_chi is None:
        r_start = 0.0
    else:
        r_start = box_center_chi - box_size[2] / 2.0

    r_planes = r_start + (jnp.arange(mesh_shape[2]) + 0.5) * dz

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





def project_flat_sky(field,
                     box_size,
                     field_size_deg,
                     field_npix,
                     distance_to_center,
                     weights=None):
    """
    Project a 3D field (Cartesian) onto a flat-sky 2D map using Born-like approximation (lightcone geometry).
    This sums the field along the line of sight, accounting for the divergence of angular coordinates with distance.

    Parameters
    ----------
    field : array [Nx, Ny, Nz]
        3D field to project.
    box_size : float or array
        Physical box size (Mpc/h).
    field_size_deg : float
        Angular field of view (degrees).
    field_npix : int
        Number of pixels for the output 2D map.
    distance_to_center : float
        Comoving distance to the center of the box (Mpc/h).
    weights : array [Nz] or None
        Optional weights to apply to each slice before summing.

    Returns
    -------
    projection : array [field_npix, field_npix]
        The projected 2D map.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    mesh_shape = field.shape
    box_size = np.atleast_1d(box_size)
    if len(box_size) == 1:
        box_size = np.repeat(box_size, 3)

    dx = box_size[0] / mesh_shape[0]
    dz = box_size[2] / mesh_shape[2]

    # Angular coordinates (radians)
    theta = np.linspace(-field_size_deg / 2, field_size_deg / 2, field_npix, endpoint=False) * (np.pi / 180.0)
    rowgrid, colgrid = np.meshgrid(theta, theta, indexing='ij')
    coords = np.stack([rowgrid, colgrid], axis=0)

    # Radial distance planes
    r_start = distance_to_center - box_size[2] / 2.0
    r_planes = r_start + (np.arange(mesh_shape[2]) + 0.5) * dz

    projection = np.zeros((field_npix, field_npix))

    for i in range(mesh_shape[2]):
        r = r_planes[i]
        # Map angular coords to pixel coords in the box at distance r
        # Gnomonic projection: x_phys = r * tan(theta)
        # x_pix = x_phys / dx + center_pix
        pixel_coords = np.tan(coords) * r / dx + mesh_shape[0] / 2.0

        # Extract slice
        plane = field[:, :, i]
        if weights is not None:
             plane = plane * weights[i]

        # Interpolate
        projected_slice = map_coordinates(plane, pixel_coords, order=1, mode='wrap')

        projection += projected_slice

    return projection


def compute_theoretical_cl_kappa(cosmo, ell, chi_min, chi_max, z_source, n_steps=100):
    """
    Compute theoretical C_l^{kappa kappa} for convergence between two comoving distance bounds.
    Uses Limber approximation and nonlinear matter power spectrum.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
    ell : array
        Ell values to compute Cl for.
    chi_min : float
        Minimum comoving distance (Mpc/h)
    chi_max : float
        Maximum comoving distance (Mpc/h)
    z_source : float
        Source redshift
    n_steps : int
        Number of integration steps

    Returns
    -------
    cl : array
        Convergence Cl values in the same shape as ell.
    """
    import jax
    import jax.numpy as jnp
    import jax_cosmo as jc
    import jax_cosmo.constants as constants

    # Source distance
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]

    # Clamp chi_max to chi_s to avoid computing P(k) at z > z_source
    chi_max = jnp.minimum(chi_max, chi_s)

    # Integration grid
    chi = jnp.linspace(chi_min, chi_max, n_steps)

    # Scale factor and redshift at integration points
    a = jc.background.a_of_chi(cosmo, chi)

    # Lensing kernel W(chi) = (3/2 Omega_m (H0/c)^2) * (chi/a) * (chi_s - chi)/chi_s
    constant_factor = 1.5 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    w_val = constant_factor * (chi / a) * (chi_s - chi) / chi_s
    w_val = jnp.where(chi < chi_s, w_val, 0.0)

    def get_cl_per_ell(ell):
        # Limber approximation: k = (ell + 0.5) / chi
        k = (ell + 0.5) / chi

        # P(k, z) element-wise along integration path
        def get_pk_elementwise(ki, ai):
            return jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))

        pk = jax.vmap(get_pk_elementwise)(k, a)

        # Integrand: W^2(chi) / chi^2 * P(k, z)
        integrand = (w_val**2 / chi**2) * pk

        # Trapezoidal integration
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    # Compute on 1D grid and interpolate
    ell = jax.lax.stop_gradient(ell)
    ell_min = jnp.min(ell)
    ell_max = jnp.max(ell)

    # Use simple linear spacing
    n_interp = 2048
    ell_interp = jnp.linspace(ell_min, ell_max, n_interp)
    ell_interp = jax.lax.stop_gradient(ell_interp)

    cl_interp = jax.vmap(get_cl_per_ell)(ell_interp)
    cl_interp = jnp.squeeze(cl_interp)
    return jnp.interp(ell, ell_interp, cl_interp)


def compute_theoretical_cl_gg(cosmo, ell, chi_min, chi_max, b1, n_steps=100):
    """
    Compute theoretical C_l^{gg} for galaxies using Limber approximation.

    Assumes galaxies are uniformly distributed between chi_min and chi_max.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
    ell : array
        Ell values to compute Cl for.
    chi_min : float
        Minimum comoving distance (Mpc/h)
    chi_max : float
        Maximum comoving distance (Mpc/h)
    b1 : float
        Linear galaxy bias
    n_steps : int
        Number of integration steps

    Returns
    -------
    cl : array
        Galaxy C_l values.
    """
    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)

    # Galaxy kernel: uniform distribution, normalized
    delta_chi = chi_max - chi_min
    w_g = b1 / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai)))(k, a)
        integrand = (w_g**2 / chi**2) * pk
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)


def compute_theoretical_cl_kg(cosmo, ell, chi_min, chi_max, z_source, b1, n_steps=100):
    """
    Compute theoretical C_l^{kappa g} cross-spectrum using Limber approximation.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
    ell : array
        Ell values to compute Cl for.
    chi_min : float
        Minimum comoving distance (Mpc/h)
    chi_max : float
        Maximum comoving distance (Mpc/h)
    z_source : float
        Source redshift for lensing
    b1 : float
        Linear galaxy bias
    n_steps : int
        Number of integration steps

    Returns
    -------
    cl : array
        Cross-spectrum C_l values.
    """
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]

    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)

    # Lensing kernel
    constant_factor = 1.5 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    w_kappa = constant_factor * (chi / a) * (chi_s - chi) / chi_s
    w_kappa = jnp.where(chi < chi_s, w_kappa, 0.0)

    # Galaxy kernel
    delta_chi = chi_max - chi_min
    w_g = b1 / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai)))(k, a)
        integrand = (w_kappa * w_g / chi**2) * pk
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)
