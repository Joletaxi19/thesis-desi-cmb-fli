"""Tests for galaxy bias and noise modeling.

These tests validate the galaxy bias functions copied from benchmark-field-level,
ensuring they work correctly in the desi_cmb_fli framework.
"""


import jax.numpy as jnp
import jax.random as jr
import numpy as np

from desi_cmb_fli.evolution import a2g
from desi_cmb_fli.galaxy_bias import (
    Planck18,
    kaiser_boost,
    kaiser_model,
    kaiser_posterior,
    lagrangian_weights,
    lin_power_mesh,
    rsd,
)
from desi_cmb_fli.initial_conditions import lin_power_mesh as init_lin_power_mesh


def planck18():
    """Create a fresh Planck18 cosmology with initialized workspace."""
    cosmo = Planck18()
    cosmo._workspace = {}
    return cosmo


def test_kaiser_boost_no_rsd():
    """Test kaiser_boost without RSD (los=None)."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    a = 0.8
    bE = 1.5

    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los=None)

    # Without RSD, boost should be just growth factor times Eulerian bias
    expected = a2g(cosmo, a) * bE
    assert jnp.isclose(boost, expected)


def test_kaiser_boost_with_rsd():
    """Test kaiser_boost with RSD."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    a = 0.8
    bE = 1.5
    los = np.array([0., 0., 1.])  # z-direction

    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)

    # boost should be a field (Fourier space), not a scalar
    assert boost.shape == (8, 8, 5)  # rfft shape
    assert jnp.all(jnp.isfinite(boost))


def test_kaiser_model_shape():
    """Test kaiser_model produces correct output shape."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    a = 0.8
    bE = 1.5

    # Generate initial field
    key = jr.PRNGKey(42)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real)

    # Apply kaiser model without RSD
    result = kaiser_model(cosmo, a, bE, init_mesh, los=None)

    # Result should be real space field with shape mesh_shape
    assert result.shape == tuple(mesh_shape)
    assert jnp.isrealobj(result)
    # Result is 1 + delta, so mean should be close to 1
    assert jnp.abs(jnp.mean(result) - 1.0) < 0.5


def test_kaiser_model_with_rsd():
    """Test kaiser_model with RSD."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    a = 0.8
    bE = 1.5
    los = np.array([0., 0., 1.])

    key = jr.PRNGKey(42)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real)

    result_no_rsd = kaiser_model(cosmo, a, bE, init_mesh, los=None)

    cosmo2 = planck18()
    result_with_rsd = kaiser_model(cosmo2, a, bE, init_mesh, los=los)

    # With RSD, result should differ
    assert not jnp.allclose(result_no_rsd, result_with_rsd)
    assert result_with_rsd.shape == tuple(mesh_shape)


def test_rsd_no_los():
    """Test RSD with los=None returns zeros."""
    cosmo = planck18()
    a = 0.8
    vel = jnp.ones((100, 3))

    dpos = rsd(cosmo, a, vel, los=None)

    assert jnp.allclose(dpos, 0.0)


def test_rsd_with_los():
    """Test RSD with line-of-sight direction."""
    cosmo = planck18()
    a = 0.8
    vel = jnp.ones((100, 3)) * 0.1  # Small velocities
    los = np.array([0., 0., 1.])

    dpos = rsd(cosmo, a, vel, los)

    # Displacement should be along los direction
    assert dpos.shape == vel.shape
    # x and y components of RSD displacement are proportional to los
    # Since los = [0,0,1], only z-component should be non-zero
    assert jnp.allclose(dpos[:, 0], 0.0)
    assert jnp.allclose(dpos[:, 1], 0.0)
    assert not jnp.allclose(dpos[:, 2], 0.0)


def test_lagrangian_weights_shape():
    """Test lagrangian_weights produces correct shape."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    box_shape = np.array([100., 100., 100.])
    a = 0.8

    # Bias parameters
    b1, b2, bs2, bn2 = 1.0, 0.0, 0.0, 0.0

    # Generate initial field
    key = jr.PRNGKey(42)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real)

    # Particle positions
    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T

    weights = lagrangian_weights(cosmo, a, pos, box_shape, b1, b2, bs2, bn2, init_mesh)

    # Weights should have one value per particle
    assert weights.shape == (pos.shape[0],)
    assert jnp.all(jnp.isfinite(weights))


def test_lagrangian_weights_b1_only():
    """Test lagrangian_weights with only linear bias."""
    cosmo = planck18()
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100., 100., 100.])
    a = 0.8

    # Only b1 non-zero
    b1, b2, bs2, bn2 = 1.5, 0.0, 0.0, 0.0

    key = jr.PRNGKey(42)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real)

    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T

    weights = lagrangian_weights(cosmo, a, pos, box_shape, b1, b2, bs2, bn2, init_mesh)

    # Mean weight should be close to 1 (since delta has zero mean)
    assert jnp.abs(jnp.mean(weights) - 1.0) < 0.3


def test_lagrangian_weights_higher_order():
    """Test lagrangian_weights with higher-order bias terms."""
    cosmo = planck18()
    mesh_shape = np.array([4, 4, 4])
    box_shape = np.array([100., 100., 100.])
    a = 0.8

    # All bias parameters non-zero
    b1, b2, bs2, bn2 = 1.5, -0.5, 0.3, -0.2

    key = jr.PRNGKey(42)
    init_mesh_real = jr.normal(key, shape=mesh_shape)
    init_mesh = jnp.fft.rfftn(init_mesh_real)

    pos = jnp.indices(mesh_shape, dtype=float).reshape(3, -1).T

    # Compute with only b1
    weights_b1 = lagrangian_weights(cosmo, a, pos, box_shape, b1, 0., 0., 0., init_mesh)

    cosmo2 = planck18()
    # Compute with all bias terms
    weights_all = lagrangian_weights(cosmo2, a, pos, box_shape, b1, b2, bs2, bn2, init_mesh)

    # Results should differ when higher-order terms are included
    assert not jnp.allclose(weights_b1, weights_all)


def test_lin_power_mesh_consistency():
    """Test that lin_power_mesh in galaxy_bias matches initial_conditions."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    box_shape = np.array([100., 100., 100.])
    a = 1.0

    # From galaxy_bias module
    pmesh_gb = lin_power_mesh(cosmo, mesh_shape, box_shape, a=a)

    # From initial_conditions module
    cosmo2 = planck18()
    pmesh_ic = init_lin_power_mesh(cosmo2, mesh_shape, box_shape, a=a)

    # Should be identical
    assert jnp.allclose(pmesh_gb, pmesh_ic)


def test_kaiser_posterior_shape():
    """Test kaiser_posterior produces correct output shapes."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    box_shape = np.array([100., 100., 100.])
    a = 0.8
    bE = 1.5
    gxy_count = 0.1  # galaxies per cell

    # Observed field in Fourier space
    key = jr.PRNGKey(42)
    delta_obs_real = jr.normal(key, shape=mesh_shape)
    delta_obs = jnp.fft.rfftn(delta_obs_real)

    means, stds = kaiser_posterior(delta_obs, cosmo, bE, a, box_shape, gxy_count, los=None)

    # Outputs should have same shape as input
    assert means.shape == delta_obs.shape
    assert stds.shape == delta_obs.shape
    assert jnp.all(jnp.isfinite(means))
    assert jnp.all(jnp.isfinite(stds))
    assert jnp.all(stds >= 0)  # Standard deviations must be non-negative


def test_kaiser_posterior_with_rsd():
    """Test kaiser_posterior with RSD."""
    cosmo = planck18()
    mesh_shape = np.array([8, 8, 8])
    box_shape = np.array([100., 100., 100.])
    a = 0.8
    bE = 1.5
    gxy_count = 0.1
    los = np.array([0., 0., 1.])

    key = jr.PRNGKey(42)
    delta_obs_real = jr.normal(key, shape=mesh_shape)
    delta_obs = jnp.fft.rfftn(delta_obs_real)

    # Without RSD
    means_no_rsd, stds_no_rsd = kaiser_posterior(delta_obs, cosmo, bE, a, box_shape, gxy_count, los=None)

    # With RSD
    cosmo2 = planck18()
    means_rsd, stds_rsd = kaiser_posterior(delta_obs, cosmo2, bE, a, box_shape, gxy_count, los=los)

    # Results should differ when RSD is included
    assert not jnp.allclose(means_no_rsd, means_rsd)
    assert not jnp.allclose(stds_no_rsd, stds_rsd)
