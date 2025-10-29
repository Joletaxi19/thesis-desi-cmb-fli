"""Unit tests for initial conditions generation."""

import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import numpy as np

from desi_cmb_fli.bricks import (
    lin_power_interp,
    lin_power_mesh,
)
from desi_cmb_fli.nbody import rfftk
from desi_cmb_fli.utils import (
    cgh2rg,
    rg2cgh,
)


def planck18():
    """Return a lightweight Planck-like cosmology for tests."""
    return jc.Cosmology(
        Omega_c=0.2607,
        Omega_b=0.0490,
        Omega_k=0.0,
        h=0.6766,
        n_s=0.9665,
        sigma8=0.8102,
        w0=-1.0,
        wa=0.0,
    )


def test_lin_power_interp_matches_jax_cosmo():
    cosmo = planck18()
    ks = jnp.logspace(-3, 0, 8)
    interp = lin_power_interp(cosmo, a=0.8, n_interp=128)
    expected = jc.power.linear_matter_power(cosmo, ks, a=0.8)
    got = interp(ks)
    assert jnp.allclose(got, expected, rtol=2e-2)


def test_lin_power_mesh_consistent_with_interp():
    cosmo = planck18()
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100.0, 100.0, 100.0])
    mesh = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.0, n_interp=64)
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki * (m / box_len)) ** 2 for ki, m, box_len in zip(kvec, mesh_shape, box_shape, strict=False)) ** 0.5
    interp = lin_power_interp(cosmo, a=1.0, n_interp=64)
    expected = interp(kmesh) * (mesh_shape / box_shape).prod()
    assert np.allclose(np.asarray(mesh), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_rg2cgh_cgh2rg_roundtrip_recovers_real_field():
    key = jr.PRNGKey(0)
    mesh = jr.normal(key, (4, 4, 4), dtype=jnp.float32)
    meshk = rg2cgh(mesh, norm="backward")
    recovered = cgh2rg(meshk, norm="backward")
    assert jnp.allclose(recovered, mesh, rtol=1e-5, atol=1e-5)
