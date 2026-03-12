"""
CMB Lensing Module - Integrated with existing field-level inference

This module provides CMB lensing observables that can be computed from
the matter density field inferred from galaxy clustering.
"""

import astropy.units as u
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
from jax.scipy.ndimage import map_coordinates

# Fraction of the flat-sky Nyquist frequency used as the hard band-limit
# for the likelihood scale cut (nyquist_mask in model.py).
NYQUIST_FRACTION = 0.5

# Gnomview oversampling factor: extract the patch at this many times the
# final resolution
GNOMVIEW_OVERSAMPLE = 8


def prepare_abacus_kappa(
    healpix_map: "np.ndarray",
    lon: float,
    lat: float,
    npix_out: int,
    field_size_deg: float,
    oversample: int = GNOMVIEW_OVERSAMPLE,
) -> "np.ndarray":
    """Degrade an AbacusSummit HEALPix κ map to the model flat-sky grid.

    Pipeline:
      1. ``ud_grade`` to nside 4096 (memory; negligible pixel-window at ℓ<600).
      2. ``hp.gnomview`` at *oversample × npix_out* (nearest-neighbour lookup).
      3. **Fourier crop**: ``fft2 → fftshift → crop central npix_out×npix_out →
         ifftshift → ifft2``.  This is a low-pass at the target Nyquist
         and a downsample in one step.  Normalised by ``(N_out/N_hd)²``.

    The scale cut at 50 % of Nyquist is **not** applied here — it belongs in
    the likelihood (``model.py``), where it is applied symmetrically to both
    the Abacus observation and the model prediction.

    Parameters
    ----------
    healpix_map : np.ndarray
        Full-sky HEALPix map (RING ordering).
    lon, lat : float
        Galactic coordinates of the patch centre (degrees).
    npix_out : int
        Pixels per side of the target (model) grid.
    field_size_deg : float
        Angular side length of the flat-sky field (degrees).
    oversample : int
        Oversampling factor for the gnomview extraction (default 8).

    Returns
    -------
    np.ndarray, shape (npix_out, npix_out)
        Band-limited flat-sky κ patch at the model resolution.
    """
    import healpy as hp
    from scipy.fft import fft2, fftshift, ifft2, ifftshift

    # 1. ud_grade for memory
    nside_in = hp.get_nside(healpix_map)
    if nside_in > 4096:
        healpix_map = hp.ud_grade(healpix_map, 4096)
        print(f"[Abacus] ud_grade {nside_in} → 4096")

    # 2. gnomview at high resolution (nearest-neighbour pixel lookup)
    npix_hd = npix_out * oversample
    reso_hd = field_size_deg * 60.0 / npix_hd   # arcmin per pixel
    patch_hd = np.asarray(hp.gnomview(
        healpix_map,
        rot=[lon, lat],
        reso=reso_hd,
        xsize=npix_hd,
        return_projected_map=True,
        no_plot=True,
    ))

    # 3. Fourier crop: exact low-pass + downsample
    F = fftshift(fft2(patch_hd))
    # Crop central npix_out × npix_out
    c = npix_hd // 2
    h = npix_out // 2
    F_crop = F[c - h : c + h, c - h : c + h]
    patch_lr = np.real(ifft2(ifftshift(F_crop))) * (npix_out / npix_hd) ** 2

    ell_nyq = 180.0 * npix_out / field_size_deg
    print(
        f"[Abacus] prepare_abacus_kappa: "
        f"gnomview {npix_hd}×{npix_hd} (reso={reso_hd:.2f}'/pix) "
        f"→ Fourier crop {npix_out}×{npix_out} "
        f"(ℓ_Nyq={ell_nyq:.0f})"
    )
    return patch_lr


def lensing_kernel(cosmo, chi, a, chi_source):
    """
    Born/Limber lensing kernel W_kappa(chi) for a source plane at chi_source.
    """
    prefactor = 1.5 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    geometry = jnp.clip((chi_source - chi) / chi_source, 0.0, jnp.inf)
    return prefactor * (chi / a) * geometry


def convergence_Born_mesh(cosmo,
                           density_field,
                           r_planes,
                           a_planes,
                           dx,
                           dz,
                           coords,
                           z_source):
    """
    Compute the Born convergence from a 3D density field with anti-aliasing.

    Per z-slice:
    1. Bilinear interpolation (map_coordinates order=1) reprojects from
       comoving to angular coordinates.
    2. FFT2 → deconvolve the bilinear sinc² window → apply per-shell
       ℓ_max = k_Nyq_3D × χ filter → accumulate in Fourier space.
    3. Final IFFT2.

    The ℓ_max filter ensures each shell only contributes angular modes
    ℓ < π/dx × χ, preventing 3D-Nyquist aliased power from leaking
    into the angular spectrum. The sinc² deconvolution corrects the
    attenuation introduced by bilinear interpolation.
    """
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    n_sources = len(z_source)
    n_planes = len(r_planes)
    nx = density_field.shape[0]
    npix = coords.shape[1]  # field_npix

    # Angular pixel spacing (radians) from coordinate grid
    theta_pix = coords[0, 1, 0] - coords[0, 0, 0]

    # ℓ grid for the output angular map
    ell_x = jnp.fft.fftfreq(npix) * (2 * jnp.pi / theta_pix)
    ell_y = jnp.fft.rfftfreq(npix) * (2 * jnp.pi / theta_pix)
    ell_x_grid, ell_y_grid = jnp.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = jnp.sqrt(ell_x_grid**2 + ell_y_grid**2)  # [npix, npix//2+1]

    # 3D Nyquist wavenumber
    k_nyq = jnp.pi / dx

    rfft_shape = (npix, npix // 2 + 1)

    def scan_fn(kappa_k, i):
        # Overdensity on this z-slice
        plane = density_field[:, :, i] - 1.0

        # Reproject from comoving to angular coordinates (bilinear)
        pixel_coords = jnp.tan(coords) * r_planes[i] / dx + nx / 2.0
        im = map_coordinates(plane, pixel_coords, order=1, mode="wrap")

        # Lensing kernel weight: shape [n_sources, 1, 1]
        kernel = lensing_kernel(cosmo, r_planes[i], a_planes[i], r_s).reshape([-1, 1, 1])
        im_weighted = im * dz * kernel  # [n_sources, npix, npix]

        # To Fourier space
        im_k = jnp.fft.rfft2(im_weighted)  # [n_sources, npix, npix//2+1]

        # Deconvolve bilinear interpolation: transfer = sinc²(ℓ dx / (2π χ))
        chi = jnp.maximum(r_planes[i], 1.0)
        sinc_arg = ell_grid * dx / (2 * jnp.pi * chi)
        W_bilinear = jnp.sinc(sinc_arg) ** 2  # jnp.sinc(x) = sin(πx)/(πx)
        im_k = im_k / jnp.maximum(W_bilinear, 1e-6)

        # Anti-aliasing: zero modes beyond ℓ_max(z) = k_Nyq × χ
        ell_max = k_nyq * chi
        im_k = im_k * (ell_grid <= ell_max)

        return kappa_k + im_k, None

    init_kappa_k = jnp.zeros((n_sources,) + rfft_shape, dtype=complex)
    kappa_k, _ = jax.lax.scan(scan_fn, init_kappa_k, jnp.arange(n_planes))

    kappa = jnp.fft.irfft2(kappa_k, s=(npix, npix))

    return kappa


def density_field_to_convergence(density_field,
                                 box_size,
                                 cosmo,
                                 field_size_deg,
                                 field_npix,
                                 z_source=1.0,
                                 box_center_chi=None):
    """
    Convert a 3D density field to a CMB convergence map via mesh-based Born projection.

    Uses bilinear interpolation (map_coordinates order=1) per z-slice to reproject
    from comoving to angular coordinates, matching the jax-fli approach.

    Parameters
    ----------
    density_field : array [Nx, Ny, Nz]
        3D density field (1+delta) from FieldLevelModel.
    box_size : array [3]
        Physical box size in Mpc/h.
    cosmo : jax_cosmo.Cosmology
        Cosmology object.
    field_size_deg : float
        Angular size of the field in degrees.
    field_npix : int
        Number of pixels for the convergence map.
    z_source : float or array
        Source redshift(s).
    box_center_chi : float, optional
        Comoving distance to the center of the box (Mpc/h).

    Returns
    -------
    kappa : array [field_npix, field_npix]
        Convergence map.
    """
    mesh_shape_local = density_field.shape
    dx = box_size[0] / mesh_shape_local[0]
    dz = box_size[2] / mesh_shape_local[2]

    if box_center_chi is None:
        r_start = 0.0
    else:
        r_start = box_center_chi - box_size[2] / 2.0

    r_planes = r_start + (jnp.arange(mesh_shape_local[2]) + 0.5) * dz
    a_planes = jc.background.a_of_chi(cosmo, r_planes)

    rowgrid, colgrid = np.meshgrid(
        np.linspace(-field_size_deg / 2, field_size_deg / 2, field_npix, endpoint=False),
        np.linspace(-field_size_deg / 2, field_size_deg / 2, field_npix, endpoint=False),
        indexing='ij'
    )
    coords = jnp.array(np.stack([rowgrid, colgrid], axis=0)) * u.deg.to(u.rad)

    z_source_array = jnp.atleast_1d(z_source)
    is_scalar = jnp.ndim(z_source) == 0

    kappa = convergence_Born_mesh(
        cosmo, density_field, r_planes, a_planes, dx, dz, coords, z_source_array,
    )

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
    Project a 3D field (Cartesian) onto a flat-sky 2D map with anti-aliasing.

    Per z-slice: bilinear interpolation reprojects from comoving to angular
    coordinates, then in Fourier space the bilinear sinc² window is deconvolved
    and a per-shell ℓ_max = k_Nyq × χ filter removes 3D-Nyquist aliased power.
    The accumulation is done in Fourier space; a final IFFT2 produces the map.

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

    # Angular pixel spacing (radians)
    theta_pix = theta[1] - theta[0]

    # ℓ grid for anti-aliasing
    ell_x = np.fft.fftfreq(field_npix) * (2 * np.pi / theta_pix)
    ell_y = np.fft.rfftfreq(field_npix) * (2 * np.pi / theta_pix)
    ell_x_grid, ell_y_grid = np.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = np.sqrt(ell_x_grid**2 + ell_y_grid**2)

    # 3D Nyquist wavenumber
    k_nyq = np.pi / dx

    # Radial distance planes
    r_start = distance_to_center - box_size[2] / 2.0
    r_planes = r_start + (np.arange(mesh_shape[2]) + 0.5) * dz

    rfft_shape = (field_npix, field_npix // 2 + 1)
    proj_k = np.zeros(rfft_shape, dtype=complex)

    for i in range(mesh_shape[2]):
        r = r_planes[i]
        pixel_coords = np.tan(coords) * r / dx + mesh_shape[0] / 2.0

        plane = field[:, :, i]
        if weights is not None:
            plane = plane * weights[i]

        projected_slice = map_coordinates(plane, pixel_coords, order=1, mode='wrap')
        slice_k = np.fft.rfft2(projected_slice)

        # Deconvolve bilinear interpolation: sinc²(ℓ dx / (2π χ))
        chi = max(r, 1.0)
        sinc_arg = ell_grid * dx / (2 * np.pi * chi)
        W_bilinear = np.sinc(sinc_arg) ** 2
        slice_k /= np.maximum(W_bilinear, 1e-6)

        # Anti-aliasing: zero modes beyond ℓ_max(z) = k_Nyq × χ
        ell_max = k_nyq * chi
        slice_k *= (ell_grid <= ell_max)

        proj_k += slice_k

    projection = np.fft.irfft2(proj_k, s=(field_npix, field_npix))
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

    # Source distance
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]

    # Clamp chi_max to chi_s to avoid computing P(k) at z > z_source
    chi_max = jnp.minimum(chi_max, chi_s)

    # Integration grid
    chi = jnp.linspace(chi_min, chi_max, n_steps)

    # Scale factor and redshift at integration points
    a = jc.background.a_of_chi(cosmo, chi)

    w_val = lensing_kernel(cosmo, chi, a, chi_s)

    def get_cl_per_ell(ell):
        # Limber approximation: k = (ell + 0.5) / chi
        k = (ell + 0.5) / chi

        # P(k, z) element-wise along integration path
        def get_pk_elementwise(ki, ai):
            return jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai))

        pk = jax.vmap(get_pk_elementwise)(k, a)

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


def compute_theoretical_cl_gg(cosmo, ell, chi_min, chi_max, b1, n_steps=100, bias_of_a=None, k_nyq=None):
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
    k_nyq : float, optional
        3D Nyquist wavenumber (π/Δx, h/Mpc).  When provided, shells with
        χ < (ℓ+0.5)/k_nyq are excluded from the integral, matching the
        ℓ_max = k_nyq × χ resolution cap applied by project_flat_sky.

    Returns
    -------
    cl : array
        Galaxy C_l values.
    """
    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)

    delta_chi = chi_max - chi_min
    if bias_of_a is None:
        bias = jnp.ones_like(a) * b1
    else:
        bias = bias_of_a(a)
    w_g = bias / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai)))(k, a)
        integrand = (w_g**2 / chi**2) * pk
        if k_nyq is not None:
            # Exclude shells where the required k_⊥ = (ℓ+0.5)/χ exceeds k_Nyq
            integrand = integrand * (chi >= (ell_val + 0.5) / k_nyq)
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)


def compute_theoretical_cl_kg(cosmo, ell, chi_min, chi_max, z_source, b1, n_steps=100, bias_of_a=None, k_nyq=None):
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
    k_nyq : float, optional
        3D Nyquist wavenumber (π/Δx, h/Mpc).  When provided, shells with
        χ < (ℓ+0.5)/k_nyq are excluded from the integral, matching the
        ℓ_max = k_nyq × χ resolution cap applied by project_flat_sky.

    Returns
    -------
    cl : array
        Cross-spectrum C_l values.
    """
    chi_s = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]

    chi = jnp.linspace(chi_min, chi_max, n_steps)
    a = jc.background.a_of_chi(cosmo, chi)

    w_kappa = lensing_kernel(cosmo, chi, a, chi_s)

    delta_chi = chi_max - chi_min
    if bias_of_a is None:
        bias = jnp.ones_like(a) * b1
    else:
        bias = bias_of_a(a)
    w_g = bias / delta_chi

    def get_cl_per_ell(ell_val):
        k = (ell_val + 0.5) / chi
        pk = jax.vmap(lambda ki, ai: jnp.squeeze(jc.power.nonlinear_matter_power(cosmo, ki, ai)))(k, a)
        integrand = (w_kappa * w_g / chi**2) * pk
        if k_nyq is not None:
            # Exclude shells where the required k_⊥ = (ℓ+0.5)/χ exceeds k_Nyq
            integrand = integrand * (chi >= (ell_val + 0.5) / k_nyq)
        dx = chi[1] - chi[0]
        return (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * dx

    return jax.vmap(get_cl_per_ell)(ell)


def compute_cl_high_z(cosmo, ell_grid, chi_min, chi_max, z_source,
                      mode="fixed",
                      cosmo_fid=None,
                      cl_cached=None,
                      gradients=None,
                      loc_fid=None,
                      n_steps=100):
    """
    Compute high-z C_l^{kappa kappa} correction with multiple strategies.

    This function centralizes the logic for computing the high-redshift contribution
    to the lensing power spectrum. Three modes are supported:

    - 'fixed': Uses pre-computed C_l at fiducial cosmology (fastest, least accurate)
    - 'taylor': Applies first-order Taylor expansion around fiducial (fast, accurate for small deviations)
    - 'exact': Computes dynamically for current cosmology (slow, most accurate)

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Current cosmology for which to compute C_l
    ell_grid : array
        2D grid of ell values (typically from FFT grid)
    chi_min : float
        Minimum comoving distance for integration (Mpc/h)
    chi_max : float
        Maximum comoving distance for integration (Mpc/h)
    z_source : float
        Source redshift for lensing
    mode : str, optional
        Computation mode: 'fixed', 'taylor', or 'exact' (default: 'fixed')
    cosmo_fid : jax_cosmo.Cosmology, optional
        Fiducial cosmology (required for 'taylor' mode)
    cl_cached : array, optional
        Pre-computed C_l at fiducial cosmology (required for 'fixed' and 'taylor' modes)
    gradients : dict, optional
        Dictionary with 'dCl_dOm' and 'dCl_ds8' arrays (required for 'taylor' mode)
    loc_fid : dict, optional
        Fiducial parameter values with 'Omega_m' and 'sigma8' keys (required for 'taylor' mode)
    n_steps : int, optional
        Number of integration steps for 'exact' mode (default: 100)

    Returns
    -------
    cl_high_z : array
        High-z C_l correction, same shape as ell_grid

    Raises
    ------
    ValueError
        If required parameters for the selected mode are not provided

    Examples
    --------
    Fixed mode (fastest):
    >>> cl = compute_cl_high_z(cosmo, ell_grid, 320, 14000, 1100,
    ...                        mode='fixed', cl_cached=cl_fid)

    Taylor mode (accurate for small deviations):
    >>> cl = compute_cl_high_z(cosmo, ell_grid, 320, 14000, 1100,
    ...                        mode='taylor', cl_cached=cl_fid,
    ...                        gradients={'dCl_dOm': grad_om, 'dCl_ds8': grad_s8},
    ...                        loc_fid={'Omega_m': 0.3, 'sigma8': 0.8})

    Exact mode (most accurate, slowest):
    >>> cl = compute_cl_high_z(cosmo, ell_grid, 320, 14000, 1100,
    ...                        mode='exact')
    """
    if mode == "fixed":
        if cl_cached is None:
            raise ValueError("mode='fixed' requires cl_cached parameter")
        return cl_cached

    elif mode == "taylor":
        if cl_cached is None or gradients is None or loc_fid is None:
            raise ValueError("mode='taylor' requires cl_cached, gradients, and loc_fid parameters")

        # Extract current cosmological parameters
        Om_curr = cosmo.Omega_c + cosmo.Omega_b
        s8_curr = cosmo.sigma8

        # Compute deviations from fiducial
        dOm = Om_curr - loc_fid["Omega_m"]
        ds8 = s8_curr - loc_fid["sigma8"]

        # Apply first-order Taylor correction
        cl_high_z = cl_cached + gradients['dCl_dOm'] * dOm + gradients['dCl_ds8'] * ds8

        # Ensure non-negative
        return jnp.maximum(cl_high_z, 1e-30)

    elif mode == "exact":
        # Compute chi_source for current cosmology
        chi_source = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_source))[0]

        # Use chi_source if chi_max not provided
        if chi_max is None:
            chi_max = chi_source

        # Compute C_l dynamically (expensive!)
        return compute_theoretical_cl_kappa(
            cosmo,
            ell_grid,
            chi_min,
            chi_max,
            z_source,
            n_steps=n_steps
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'fixed', 'taylor', or 'exact'")


def load_abacus_observation(abacus_cfg: dict, model, smooth: bool = True) -> dict:
    """
    Load an external AbacusSummit κ patch and build an observed κ map.

    Performs a gnomonic (flat-sky) projection of the AbacusSummit HEALPix
    convergence map onto the model grid, then adds a synthetic reconstruction
    noise realization drawn from model.nell_grid.

    The noise realization is generated in Fourier space so that its flat-sky
    power spectrum matches N_ell exactly.

    Parameters
    ----------
    abacus_cfg : dict
        Sub-dict from config (typically cfg["abacus_kappa"]) with optional keys:
            file (str)         : Path to the ASDF file (default: AbacusSummit c000 ph000).
            lon, lat (float)   : Galactic centre of the gnomonic patch (default: 75°, 20°).
            noise_seed (int)   : JAX PRNGKey seed for the noise draw.
                                 Default is 7777.
    model : FieldLevelModel
        Initialized model.  Required attributes:
            cmb_enabled, cmb_field_size_deg, cmb_field_npix, nell_grid.
    smooth : bool, optional
        If True (default), band-limit the HEALPix map to the simulation pixel
        scale (Gaussian beam, FWHM = reso_arcmin) before gnomonic projection.
        Disabling this retains small-scale power up to ell ≈ 2×nside, which
        systematically biases σ_8 high — only set False for diagnostics.

    Returns
    -------
    dict with keys:
        'kappa_obs'  : 2-D JAX array, κ_Abacus + noise (shape: Npix × Npix).
        'kappa_pred' : 2-D JAX array, noiseless κ_Abacus patch (full LOS).
    """
    if not model.cmb_enabled:
        raise ValueError(
            "load_abacus_observation requires model.cmb_lensing.enabled=true"
        )

    # ── Validate optional dependencies are available ─────────────────────────
    try:
        import healpy as hp
    except ImportError as exc:
        raise RuntimeError(
            "healpy is required for observation_mode='abacus'. "
            "Install it with: conda install -c conda-forge healpy"
        ) from exc
    try:
        import asdf as _asdf
    except ImportError as exc:
        raise RuntimeError(
            "asdf is required for observation_mode='abacus'. "
            "Install it with: pip install asdf"
        ) from exc

    # ── Config ───────────────────────────────────────────────────────────────
    _DEFAULT_FILE = (
        "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/"
        "v1/AbacusSummit_base_c000_ph000/kappa_00047.asdf"
    )
    abacus_file = abacus_cfg.get("file", _DEFAULT_FILE)
    lon_patch   = float(abacus_cfg.get("lon", 75.0))
    lat_patch   = float(abacus_cfg.get("lat", 20.0))
    noise_seed  = int(abacus_cfg.get("noise_seed", 7777))

    if "noise_seed" not in abacus_cfg:
        print(
            "[Abacus] WARNING: noise_seed not set in abacus_kappa config. "
            "Using default seed=7777.  All runs without an explicit noise_seed "
            "will produce the same noise realisation — set it if you need "
            "independent draws."
        )

    npix        = model.cmb_field_npix
    field_deg   = model.cmb_field_size_deg
    reso_arcmin = field_deg * 60.0 / npix

    # ── Load HEALPix map and project (~12 GB peak RAM) ───────────────────────
    print(f"[Abacus] Loading {abacus_file} ...")
    _kappa_hp = None
    try:
        with _asdf.open(abacus_file) as _f:
            _kappa_hp = np.array(_f.tree["data"]["kappa"])

        if smooth:
            _patch_np = prepare_abacus_kappa(
                _kappa_hp, lon_patch, lat_patch, npix, field_deg,
            )
        else:
            print("[Abacus] Skipping band-limit (smooth=False) — diagnostic mode.")
            _patch_np = np.asarray(hp.gnomview(
                _kappa_hp,
                rot=[lon_patch, lat_patch],
                reso=reso_arcmin,
                xsize=npix,
                return_projected_map=True,
                no_plot=True,
            ))
    finally:
        del _kappa_hp   # free ~12 GB

    # ── Validate coverage ──────────────────────────────────────────────────
    # AbacusSummit uses 0 (not hp.UNSEEN) as sentinel for uncovered pixels.
    n_zero = int(np.sum(_patch_np == 0.0))
    zero_frac = n_zero / _patch_np.size
    if zero_frac > 0.001:
        raise ValueError(
            f"[Abacus] {n_zero} pixels ({zero_frac:.2%}) are exactly zero — "
            "likely outside the AbacusSummit lightcone footprint. "
            "Reduce the field size (box_shape Lx/Ly) or adjust abacus_kappa.lon / .lat. "
            f"Current field: {field_deg:.1f}° (requires full lightcone coverage)."
        )

    # Print diagnostic stats before converting ─────────────────────────────
    print("[Abacus] Patch stats (noiseless):")
    print(f"  shape  = {_patch_np.shape}")
    print(f"  min    = {float(_patch_np.min()):.6f}")
    print(f"  max    = {float(_patch_np.max()):.6f}")
    print(f"  mean   = {float(_patch_np.mean()):.6f}")
    print(f"  std    = {float(_patch_np.std()):.6f}")
    print(f"  median = {float(np.median(_patch_np)):.6f}")
    print(f"  zeros  = {n_zero} ({zero_frac:.4%})")

    _patch = jnp.array(_patch_np)
    del _patch_np

    # ── Add reconstruction noise realization ─────────────────────────────────
    field_area_sr = (field_deg * np.pi / 180.0) ** 2
    var_k  = model.nell_grid * (npix ** 4 / field_area_sr)
    var_k  = jnp.where(jnp.isinf(var_k) | jnp.isnan(var_k), 0.0, var_k)

    eps     = jr.normal(jr.key(noise_seed), shape=(npix, npix))
    eps_k   = jnp.fft.fftn(eps)
    noise_k = eps_k * jnp.sqrt(var_k) / npix
    noise_k = noise_k.at[0, 0].set(0.0)
    noise_map = jnp.fft.ifftn(noise_k).real

    kappa_obs = _patch + noise_map

    print(
        f"[Abacus] Noise added (seed={noise_seed}): "
        f"std(signal)={float(jnp.std(_patch)):.4f}, "
        f"std(noise)={float(jnp.std(noise_map)):.4f}, "
        f"std(total)={float(jnp.std(kappa_obs)):.4f}"
    )
    del noise_map

    truth = {"kappa_obs": kappa_obs, "kappa_pred": _patch}

    print(
        f"[Abacus] Observation ready "
        f"(lon={lon_patch}, lat={lat_patch}), "
        f"shape={truth['kappa_obs'].shape}, "
        f"std={float(jnp.std(truth['kappa_obs'])):.4f}"
    )
    return truth
