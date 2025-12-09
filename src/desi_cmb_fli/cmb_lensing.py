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


def compute_signal_capture_ratio(cosmo, kappa_map, field_size_deg, npix, z_source):
    """
    Compute the ratio R_ell = C_ell^box / C_ell^total robustly.
    Handles small fields of view by adapting bin sizes.
    """
    import jax.numpy as jnp
    import jax_cosmo as jc
    import numpy as np

    # 1. Setup Geometry
    field_size_rad = field_size_deg * np.pi / 180.

    # Fundamental frequency (separation between Fourier modes)
    delta_ell = 2 * np.pi / field_size_rad

    # 2. Compute 2D Power Spectrum of the Box
    kappa_k = np.fft.fftn(kappa_map)
    P_k_2d = np.abs(kappa_k)**2 * (field_size_rad**2 / npix**4)

    # Grid of ell values
    ell_freq = np.fft.fftfreq(npix, d=field_size_rad/npix) * 2 * np.pi
    lx, ly = np.meshgrid(ell_freq, ell_freq, indexing='ij')
    ell_grid = np.sqrt(lx**2 + ly**2)

    # 3. ROBUST BINNING (The Fix)
    # We ensure bins are at least as wide as delta_ell to avoid empty bins
    # and "aliasing" in the 1D estimation.
    bin_width = max(50.0, delta_ell * 1.1)
    ell_max = ell_grid.max()
    ell_edges = np.arange(0, ell_max + bin_width, bin_width)

    # Binning logic
    dig = np.digitize(ell_grid.flatten(), ell_edges)
    mask = (dig > 0) & (dig < len(ell_edges))

    # ell_flat = ell_grid.flatten() # Unused
    P_flat = P_k_2d.flatten()

    bin_counts = np.bincount(dig[mask], minlength=len(ell_edges))
    bin_sums = np.bincount(dig[mask], weights=P_flat[mask], minlength=len(ell_edges))

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        C_ell_box_1d = bin_sums / bin_counts

    bin_centers = 0.5 * (ell_edges[1:] + ell_edges[:-1])

    # Filter valid bins
    valid = (bin_counts[1:] > 0) & np.isfinite(C_ell_box_1d[1:])
    # Note: bin_counts[0] corresponds to values below ell_edges[0]=0, usually empty or DC
    # The output of bincount has length len(ell_edges).
    # Index i in bincount corresponds to bin i (digitize output).
    # Since we masked dig > 0, we look at indices 1 to end.

    # Align arrays
    C_ell_box_1d = C_ell_box_1d[1:]
    current_bin_centers = bin_centers

    # Apply validity mask
    C_ell_box_1d = C_ell_box_1d[valid]
    current_bin_centers = current_bin_centers[valid]

    # 4. Theory C_ell
    # We calculate theory at the exact bin centers to minimize interpolation errors
    probe = jc.probes.WeakLensing([jc.redshift.delta_nz(z_source)], sigma_e=0)
    cl_theory = jc.angular_cl.angular_cl(cosmo, current_bin_centers, [probe])
    cl_theory = jnp.squeeze(cl_theory)

    # 5. Compute Ratio in 1D first (More stable)
    # Ratio at bin centers
    ratio_1d = C_ell_box_1d / cl_theory

    # 6. Interpolate Ratio to 2D
    # Using 'nearest' or linear extrapolation for low-ell is safer than left=0
    # because the ratio should be roughly constant at low ell (scale independent growth)
    ratio_2d = np.interp(ell_grid.flatten(), current_bin_centers, ratio_1d, left=ratio_1d[0], right=0)
    ratio_2d = ratio_2d.reshape(npix, npix)

    # Cap
    ratio_2d = np.clip(ratio_2d, 0, 1.0)

    return ratio_2d
