"""Generate initial conditions for cosmological simulations.

This module provides tools to generate Gaussian random fields in 3D boxes,
following a specified cosmological power spectrum. These initial conditions
serve as the starting point for gravitational evolution in field-level inference.

The implementation uses JAX for GPU acceleration and jax_cosmo for cosmological
power spectra.
"""

import jax_cosmo as jc
import numpy as np
from jax import numpy as jnp
from jax_cosmo import Cosmology
from jaxpm.kernels import longrange_kernel
from jaxpm.painting import cic_paint, cic_read
from scipy.special import legendre


def rfftk(shape):
    """
    Return wavevectors in cell units for rfftn.
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.rfftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kx, ky, kz


def lin_power_interp(cosmo=Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    ks = jnp.logspace(-4, 1, n_interp)
    logpows = jnp.log(jc.power.linear_matter_power(cosmo, ks, a=a))
    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    def pow_fn(x):
        return jnp.exp(jnp.interp(x.reshape(-1), ks, logpows, left=-jnp.inf, right=-jnp.inf)).reshape(x.shape)
    # pows = jc.power.linear_matter_power(cosmo, ks, a=a)
    # pow_fn = lambda x: jnp.interp(x.reshape(-1), ks, pows, left=0., right=0.).reshape(x.shape)
    return pow_fn


def lin_power_mesh(cosmo:Cosmology, mesh_shape, box_shape, a=1., n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pow_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki  * (m / box_len))**2 for ki, m, box_len in zip(kvec, mesh_shape, box_shape, strict=False))**0.5
    return pow_fn(kmesh) * (mesh_shape / box_shape).prod() # from [Mpc/h]^3 to cell units


def r2chshape(shape):
    """
    Real shape to complex Hermitian shape.
    """
    return (*shape[:2], shape[2]//2+1)


def _rg2cgh(mesh, part="real", norm="backward"):
    """
    Return the real and imaginary parts of a complex Gaussian Hermitian tensor
    obtained by permuting and reweighting a real Gaussian tensor (3D).
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(mesh.shape)
    assert np.all(shape % 2 == 0), "dimension lengths must be even."

    hx, hy, hz = shape // 2
    meshk = jnp.zeros(r2chshape(shape))

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    meshk = meshk.at[:, :, 1:-1].set(mesh[:, :, sliz])

    for k in [0, hz]:  # two faces
        meshk = meshk.at[:, 1:hy, k].set(mesh[:, sliy, k])
        meshk = meshk.at[1:, hy + 1:, k].set(mesh[1:, sliy, k][::-1, ::-1])
        meshk = meshk.at[0, hy + 1:, k].set(mesh[0, sliy, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            meshk = meshk.at[:, hy + 1:, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            meshk = meshk.at[1:hx, j, k].set(mesh[slix, j, k])
            meshk = meshk.at[hx + 1:, j, k].set(mesh[slix, j, k][::-1])
            if part == "imag" and norm != "amp":
                meshk = meshk.at[hx + 1:, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    meshk = meshk.at[i, j, k].set(mesh[i, j, k])
                    if norm != "amp":
                        meshk = meshk.at[i, j, k].multiply(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        meshk /= (2 / shape.prod())**.5
    elif norm == "forward":
        meshk /= (2 * shape.prod())**.5
    elif norm == "ortho":
        meshk /= 2**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return meshk

def rg2cgh(mesh, norm="backward"):
    """
    Permute and reweight a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _rg2cgh(mesh, part="real", norm=norm)
    if norm == "amp":
        return real
    else:
        imag = _rg2cgh(mesh, part="imag", norm=norm)
        return real + 1j * imag

def safe_div(x, y):
    """
    Safe division, where division by zero is zero.
    Uses the "double-where" trick for safe gradient,
    see https://github.com/jax-ml/jax/issues/5039
    """
    y_nozeros = jnp.where(y==0, 1, y)
    return jnp.where(y==0, 0, x / y_nozeros)

def ch2rshape(kshape):
    """
    Complex Hermitian shape to real shape.

    Assume last real shape is even to lift the ambiguity.
    """
    return (*kshape[:2], 2*(kshape[2]-1))

def _cgh2rg(meshk, part="real", norm="backward"):
    """
    Return the "real" and "imaginary" partitions of a real Gaussian tensor (3D)
    obtained by permuting and reweighting the real or imaginary parts of a
    complex Gaussian Hermitian tensor.
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(ch2rshape(meshk.shape))
    assert np.all(shape % 2 == 0), "dimension lengths must be even."

    hx, hy, hz = shape // 2
    mesh = jnp.zeros(shape)

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    mesh = mesh.at[:, :, sliz].set(meshk[:, :, 1:-1])

    for k in [0, hz]:  # two faces
        mesh = mesh.at[:, sliy, k].set(meshk[:, 1:hy, k])
        mesh = mesh.at[1:, sliy, k].set(meshk[1:, hy + 1:, k][::-1, ::-1])
        mesh = mesh.at[0, sliy, k].set(meshk[0, hy + 1:, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            mesh = mesh.at[:, sliy, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            mesh = mesh.at[slix, j, k].set(meshk[1:hx, j, k])
            mesh = mesh.at[slix, j, k].set(meshk[hx + 1:, j, k][::-1])
            if part == "imag" and norm != "amp":
                mesh = mesh.at[slix, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    mesh = mesh.at[i, j, k].set(meshk[i, j, k])
                    if norm != "amp":
                        mesh = mesh.at[i, j, k].divide(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        mesh *= (2 / shape.prod())**.5
    elif norm == "forward":
        mesh *= (2 * shape.prod())**.5
    elif norm == "ortho":
        mesh *= 2**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return mesh


def cgh2rg(meshk, norm="backward"):
    """
    Permute and reweight a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _cgh2rg(meshk.real, part="real", norm=norm)
    if norm == "amp":
        # Give same amplitude to wavevector real and imaginary part
        imag = _cgh2rg(meshk.real, part="imag", norm=norm)
    else:
        imag = _cgh2rg(meshk.imag, part="imag", norm=norm)
    return real + imag


def samp2base_mesh(init:dict, precond=False, transfer=None, inv=False, temp=1.) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevector coefficients at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name+'_' if inv else in_name[:-1]
        transfer *= temp**.5

        # Reparametrize
        if not inv:
            if precond=='direct':
                # Sample in direct space
                mesh = jnp.fft.rfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                # Sample in fourier space
                mesh = rg2cgh(mesh)

            mesh *= transfer # ~ CN(0, P)
        else:
            mesh = safe_div(mesh, transfer)

            if precond=='direct':
                mesh = jnp.fft.irfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                mesh = cgh2rg(mesh)

        return {out_name:mesh}
    return {}


def _waves(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        * If None, set dk to twice the minimum.
        * If int, specifies number of edges.
        * If float, specifies dk.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    kedges : ndarray
        Edges of the bins.
    kmesh : ndarray
        Wavenumber mesh.
    mumesh : ndarray
        Cosine mesh.
    rfftw : ndarray
        RFFT weights accounting for Hermitian symmetry.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape) # = knyquist

    if isinstance(kedges, type(None) | int | float):
        if kedges is None:
            dk = 2*np.pi / np.min(box_shape) * 2 # twice the fundamental wavenumber
        if isinstance(kedges, int):
            dk = kmax / kedges # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(0, kmax, dk) + dk/2 # from dk/2 to kmax-dk/2

    kvec = rfftk(mesh_shape) # cell units
    kvec = [ki * (m / b) for ki, m, b in zip(kvec, mesh_shape, box_shape, strict=False)] # h/Mpc physical units
    kmesh = sum(ki**2 for ki in kvec)**0.5

    if los is None:
        mumesh = 0.
    else:
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)

    rfftw = np.full_like(kmesh, 2)
    rfftw[..., 0] = 1
    if mesh_shape[-1] % 2 == 0:
        rfftw[..., -1] = 1

    return kedges, kmesh, mumesh, rfftw

def paint_kernel(kvec, order:int=2):
    """
    Compute painting kernel of given order.

    Parameters
    ----------
    kvec: list
        List of wavevectors
    order: int
        order of the kernel
        * 0: Dirac
        * 1: Nearest Grid Point (NGP)
        * 2: Cloud-In-Cell (CIC)
        * 3: Triangular-Shape Cloud (TSC)
        * 4: Piecewise-Cubic Spline (PCS)

        cf. [List and Hahn, 2024](https://arxiv.org/abs/2309.10865)

    Returns
    -------
    weights: array
        Complex kernel values
    """
    wts = [np.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    wts = (wts[0] * wts[1] * wts[2])**order
    return wts

def invlaplace_kernel(kvec, fd=False):
    """
    Compute the inverse Laplace kernel.

    Parameters
    -----------
    kvec: list
        List of wavevectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    return - safe_div(1, kk)

def gradient_kernel(kvec, direction:int, fd=False):
    """
    Compute the gradient kernel in the given direction

    Parameters
    -----------
    kvec: list
        List of wavevectors
    direction: int
        Index of the direction in which to take the gradient
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    ki = kvec[direction]
    if fd:
        ki = (8. * np.sin(ki) - np.sin(2. * ki)) / 6.
    return 1j * ki

def pm_forces(pos, mesh_shape, mesh=None, grad_fd=False, lap_fd=False, r_split=0):
    """
    Compute gravitational forces on particles using a PM scheme
    """
    if mesh is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    # elif jnp.isrealobj(mesh):
    #     delta_k = jnp.fft.rfftn(mesh)
    else:
        delta_k = mesh

    # Compute gravitational potential
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd) * longrange_kernel(kvec, r_split=r_split)

    # # If painted field, double deconvolution to account for both painting and reading
    # if mesh is None:
    #     print("deconv")
    #     pot_k /= paint_kernel(kvec, order=2)**2

    # Compute gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i, grad_fd) * pot_k), pos)
                      for i in range(3)], axis=-1)


def spectrum(mesh, mesh2=None, box_shape=None, kedges:int|float|list=None,
             comp=(0, 0), poles=0, los:np.ndarray=None):
    """
    Compute the auto and cross spectrum of 3D fields, with multipole.
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    if los is not None:
        los = np.asarray(los)
        los /= np.linalg.norm(los)
    pls = np.atleast_1d(poles)

    # FFTs and deconvolution
    if isinstance(comp, int):
        comp = (comp, comp)

    mesh = jnp.fft.rfftn(mesh, norm='ortho')
    kvec = rfftk(mesh_shape) # cell units
    mesh /= paint_kernel(kvec, order=comp[0])

    if mesh2 is None:
        mmk = mesh.real**2 + mesh.imag**2
    else:
        mesh2 = jnp.fft.rfftn(mesh2, norm='ortho')
        mesh2 /= paint_kernel(kvec, order=comp[1])
        mmk = mesh * mesh2.conj()

    # Binning
    kedges, kmesh, mumesh, rfftw = _waves(mesh_shape, box_shape, kedges, los)
    n_bins = len(kedges) + 1
    dig = np.digitize(kmesh.reshape(-1), kedges)

    # Count wavenumber in bins
    kcount = np.bincount(dig, weights=rfftw.reshape(-1), minlength=n_bins)
    kcount = kcount[1:-1]

    # Average wavenumber values in bins
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(dig, weights=(kmesh * rfftw).reshape(-1), minlength=n_bins)
    kavg = kavg[1:-1] / kcount

    # Average wavenumber power in bins
    pow = jnp.empty((len(pls), n_bins))
    for i_ell, ell in enumerate(pls):
        weights = (mmk * (2*ell+1) * legendre(ell)(mumesh) * rfftw).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else:
            # NOTE: bincount is really slow with complex numbers, so bincount real and imag parts
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2)**.5
        pow = pow.at[i_ell].set(psum)
    pow = pow[:,1:-1] / kcount * (box_shape / mesh_shape).prod() # from cell units to [Mpc/h]^3

    # kpow = jnp.concatenate([kavg[None], pk])
    if poles==0:
        return kavg, pow[0]
    else:
        return kavg, pow
