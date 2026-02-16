from functools import partial

import jax_cosmo as jc
import numpy as np
from jax import numpy as jnp
from jax_cosmo import Cosmology
from jaxpm.painting import cic_read

from desi_cmb_fli.nbody import a2f, a2g, chi2a, invlaplace_kernel, rfftk
from desi_cmb_fli.utils import cgh2rg, ch2rshape, rg2cgh, safe_div, std2trunc, trunc2std

# [Planck2015 XIII](https://arxiv.org/abs/1502.01589) Table 4 final column (best fit)
Planck15 = partial(
    Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,
)

# [Planck 2018 VI](https://arxiv.org/abs/1807.06209) Table 2 final column (best fit)
Planck18 = partial(
    Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)


def lin_power_interp(cosmo=Cosmology, a=1.0, n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    ks = jnp.logspace(-4, 1, n_interp)
    logpows = jnp.log(jc.power.linear_matter_power(cosmo, ks, a=a))

    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    def pow_fn(x):
        return jnp.exp(
            jnp.interp(x.reshape(-1), ks, logpows, left=-jnp.inf, right=-jnp.inf)
        ).reshape(x.shape)

    return pow_fn


def lin_power_mesh(cosmo: Cosmology, mesh_shape, box_shape, a=1.0, n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pow_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    kmesh = (
        sum(
            (ki * (m / box_l)) ** 2
            for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
        )
        ** 0.5
    )
    return pow_fn(kmesh) * (mesh_shape / box_shape).prod()  # from [Mpc/h]^3 to cell units


def kaiser_posterior(delta_obs, cosmo: Cosmology, bE, a, box_shape, gxy_count, los=None):
    """
    Return posterior mean and std fields of the linear matter field (at a=1) given the observed field,
    by assuming Kaiser model. All fields are in fourier space.
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(delta_obs.shape)
    pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape)
    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)

    stds = (pmeshk / (1 + gxy_count * boost**2 * pmeshk)) ** 0.5
    # Also: stds = jnp.where(pmeshk==0., 0., pmeshk / (1 + gxy_count * evolve**2 * pmeshk))**.5
    means = stds**2 * gxy_count * boost * delta_obs
    return means, stds


def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c=cosmo["Omega_m"] - Planck18.keywords["Omega_b"], sigma8=cosmo["sigma8"])


def regular_pos(mesh_shape: tuple, ptcl_shape: tuple | None = None):
    """
    Return regularly spaced particle positions in cell coordinates.
    """
    if ptcl_shape is None:
        ptcl_shape = mesh_shape

    pos = [np.linspace(0, m, p, endpoint=False) for m, p in zip(mesh_shape, ptcl_shape, strict=False)]
    return jnp.stack(np.meshgrid(*pos, indexing="ij"), axis=-1).reshape(-1, 3)


def cell2phys_pos(pos, box_center, box_shape, mesh_shape):
    """
    Convert cell coordinates to physical coordinates in Mpc/h.
    """
    pos = pos * (box_shape / mesh_shape)
    pos = pos - box_shape / 2.0
    pos = pos + box_center
    return pos


def phys2cell_pos(pos, box_center, box_shape, mesh_shape):
    """
    Convert physical coordinates in Mpc/h to cell coordinates.
    """
    pos = pos - box_center
    pos = pos + box_shape / 2.0
    pos = pos / (box_shape / mesh_shape)
    return pos


def cell2phys_vel(vel, box_shape, mesh_shape):
    """
    Convert cell velocities to physical velocities in Mpc/h.
    """
    return vel * (box_shape / mesh_shape)


def phys2cell_vel(vel, box_shape, mesh_shape):
    """
    Convert physical velocities in Mpc/h to cell velocities.
    """
    return vel / (box_shape / mesh_shape)


def radius_mesh(box_center, box_shape, mesh_shape, curved_sky=True, los=None):
    """
    Return radial comoving distance for each mesh cell center.
    """
    pos = regular_pos(mesh_shape)
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape).reshape(tuple(mesh_shape) + (3,))

    if curved_sky:
        return jnp.linalg.norm(pos, axis=-1)

    if los is None:
        los = safe_div(box_center, jnp.linalg.norm(box_center))
    los = jnp.asarray(los) / jnp.linalg.norm(los)
    return jnp.abs((pos * los).sum(-1))


def tophysical_pos(pos, box_center, box_shape, mesh_shape, cosmo: Cosmology, a_obs=None, curved_sky=True, los=None):
    """
    Return physical positions, radial distance, line-of-sight, and scale factor.
    """
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape)
    if curved_sky:
        los_vec = safe_div(pos, jnp.linalg.norm(pos, axis=-1, keepdims=True))
        rpos = jnp.linalg.norm(pos, axis=-1, keepdims=True)
    else:
        if los is None:
            los = safe_div(box_center, jnp.linalg.norm(box_center))
        los = jnp.asarray(los) / jnp.linalg.norm(los)
        los_vec = los
        rpos = jnp.abs((pos * los).sum(-1, keepdims=True))

    if a_obs is None:
        a = chi2a(cosmo, rpos)
    else:
        a = a_obs
    return pos, rpos, los_vec, a


def tophysical_mesh(box_center, box_shape, mesh_shape, cosmo: Cosmology, a_obs=None, curved_sky=True, los=None):
    """
    Return line-of-sight and scale-factor mesh.
    """
    pos = regular_pos(mesh_shape)
    pos = cell2phys_pos(pos, box_center, box_shape, mesh_shape).reshape(tuple(mesh_shape) + (3,))

    if curved_sky:
        los_mesh = safe_div(pos, jnp.linalg.norm(pos, axis=-1, keepdims=True))
        rmesh = jnp.linalg.norm(pos, axis=-1)
    else:
        if los is None:
            los = safe_div(box_center, jnp.linalg.norm(box_center))
        los = jnp.asarray(los) / jnp.linalg.norm(los)
        los_mesh = los
        rmesh = jnp.abs((pos * los).sum(-1))

    if a_obs is None:
        a = chi2a(cosmo, rmesh)
    else:
        a = a_obs
    return los_mesh, a


def samp2base(params: dict, config, inv=False, temp=1.0) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name + "_" if inv else in_name[:-1]

        conf = config[name]
        low, high = conf.get("low", -jnp.inf), conf.get("high", jnp.inf)
        loc_fid, scale_fid = conf["loc_fid"], conf["scale_fid"]
        scale_fid *= temp**0.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                out[out_name] = std2trunc(value, loc_fid, scale_fid, low, high)
            else:
                out[out_name] = value * scale_fid + loc_fid
        else:
            if low != -jnp.inf or high != jnp.inf:
                out[out_name] = trunc2std(value, loc_fid, scale_fid, low, high)
            else:
                out[out_name] = (value - loc_fid) / scale_fid
    return out


def samp2base_mesh(init: dict, precond=False, transfer=None, inv=False, temp=1.0) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevector coefficients at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name + "_" if inv else in_name[:-1]
        transfer *= temp**0.5

        # Reparametrize
        if not inv:
            if precond == "direct":
                # Sample in direct space
                mesh = jnp.fft.rfftn(mesh)

            elif precond in ["fourier", "kaiser", "kaiser_dyn"]:
                # Sample in fourier space
                mesh = rg2cgh(mesh)

            mesh *= transfer  # ~ CN(0, P)
        else:
            mesh = safe_div(mesh, transfer)

            if precond == "direct":
                mesh = jnp.fft.irfftn(mesh)

            elif precond in ["fourier", "kaiser", "kaiser_dyn"]:
                mesh = cgh2rg(mesh)

        return {out_name: mesh}
    return {}


def lagrangian_weights(cosmo: Cosmology, a, pos, box_shape, b1, b2, bs2, bn2, init_mesh):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::

        w = 1 + b_1 \\delta + b_2 \\left(\\delta^2 - \\braket{\\delta^2}\\right) + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta
    """
    delta_k = init_mesh
    delta = jnp.fft.irfftn(delta_k)
    growths = a2g(cosmo, a)
    if jnp.ndim(growths) > 1:
        growths = growths.squeeze()
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # delta_k = delta_k * jnp.exp( - kk_box / k_nyquist**2)
    # delta = jnp.fft.irfftn(delta_k)

    mesh_shape = delta.shape
    kvec = rfftk(mesh_shape)
    kk_box = sum(
        (ki * (m / box_l)) ** 2 for ki, m, box_l in zip(kvec, mesh_shape, box_shape, strict=False)
    )  # minus laplace kernel in h/Mpc physical units

    # Init weights
    weights = 1.0

    # Apply b1, punctual term
    delta_part = cic_read(delta, pos) * growths
    weights = weights + b1 * delta_part

    # Apply b2, punctual term
    delta2_part = delta_part**2
    weights = weights + b2 * (delta2_part - delta2_part.mean())

    # Apply bshear2, non-punctual term
    pot_k = delta_k * invlaplace_kernel(kvec)

    shear2 = 0
    for i, ki in enumerate(kvec):
        # Add diagonal terms
        shear2 = shear2 + jnp.fft.irfftn(-(ki**2) * pot_k - delta_k / 3) ** 2
        for kj in kvec[i + 1 :]:
            # Add strict-up-triangle terms (counted twice)
            shear2 = shear2 + 2 * jnp.fft.irfftn(-ki * kj * pot_k) ** 2

    shear2_part = cic_read(shear2, pos) * growths**2
    weights = weights + bs2 * (shear2_part - shear2_part.mean())

    # Apply bnabla2, non-punctual term
    delta_nl = jnp.fft.irfftn(-kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos) * growths
    weights = weights + bn2 * delta_nl_part

    return weights


def rsd(cosmo: Cosmology, vel, los, a, box_shape, mesh_shape):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and growth-time integrator velocity.
    Computed with respect to scale factor(s) and line-of-sight(s).

    No RSD if los is None.
    """
    if los is None:
        return jnp.zeros_like(vel)

    los = jnp.asarray(los)
    vel = cell2phys_vel(vel, box_shape, mesh_shape)

    growth = a2g(cosmo, a) * a2f(cosmo, a)
    if jnp.ndim(growth) == 1:
        growth = growth[:, None]
    vel = vel * growth

    dpos = (vel * los).sum(-1, keepdims=True) * los
    return phys2cell_vel(dpos, box_shape, mesh_shape)


def naive_mu2_delta(delta_k, los, mesh_shape):
    """
    Return mu^2 * delta for a possibly spatially varying line-of-sight field.

    This implements n_i n_j d_i d_j nabla^{-2} delta in real space.
    """
    los = jnp.asarray(los)
    if los.ndim == 1:
        los = los / jnp.linalg.norm(los)
    else:
        los = safe_div(los, jnp.linalg.norm(los, axis=-1, keepdims=True))

    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec)

    mu2_delta = jnp.zeros(tuple(mesh_shape))
    for i, ki in enumerate(kvec):
        mu2_delta = mu2_delta + los[..., i] ** 2 * jnp.fft.irfftn(-(ki**2) * pot_k)
        for j in range(i + 1, 3):
            mu2_delta = mu2_delta + 2.0 * los[..., i] * los[..., j] * jnp.fft.irfftn(-ki * kvec[j] * pot_k)

    return mu2_delta


def optim_mu2_delta(delta_k, los, mesh_shape):
    """
    Optimized mu^2 operator placeholder.

    Uses the same expression as naive_mu2_delta for scientific correctness.
    """
    return naive_mu2_delta(delta_k, los, mesh_shape)


def kaiser_boost(cosmo: Cosmology, a, bE, mesh_shape, los: np.ndarray = None):
    """
    Return Eulerian Kaiser boost including linear growth, Eulerian linear bias, and RSD.

    No RSD if los is None.
    """
    if los is None:
        return a2g(cosmo, a) * bE
    else:
        los = jnp.asarray(los)
        los = safe_div(los, jnp.linalg.norm(los))
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec) ** 0.5  # in cell units
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)

        return a2g(cosmo, a) * (bE + a2f(cosmo, a) * mumesh**2)


def kaiser_model(cosmo: Cosmology, a, bE, init_mesh, los: np.ndarray = None):
    """
    Kaiser model, with linear growth, Eulerian linear bias, and RSD.
    """
    mesh_shape = ch2rshape(init_mesh.shape)

    # Fast path: scalar time and fixed LOS (standard snapshot Kaiser in Fourier space).
    if jnp.ndim(a) == 0 and (los is None or jnp.ndim(los) == 1):
        boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)
        return 1 + jnp.fft.irfftn(init_mesh * boost)  # 1 + delta

    delta_lin = jnp.fft.irfftn(init_mesh)
    if los is None:
        mu2_delta = jnp.zeros_like(delta_lin)
    elif jnp.ndim(los) == 1:
        los = jnp.asarray(los)
        los = safe_div(los, jnp.linalg.norm(los))
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec) ** 0.5
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los, strict=False))
        mumesh = safe_div(mumesh, kmesh)
        mu2_delta = jnp.fft.irfftn(mumesh**2 * init_mesh)
    else:
        mu2_delta = optim_mu2_delta(init_mesh, los, mesh_shape)

    delta = bE * delta_lin + a2f(cosmo, a) * mu2_delta
    delta = a2g(cosmo, a) * delta
    return 1 + delta
