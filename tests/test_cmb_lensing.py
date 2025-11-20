"""
Unit tests for CMB lensing module.

Tests convergence computation from density fields using Born approximation.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax_cosmo import Cosmology

from desi_cmb_fli.cmb_lensing import (
    density_field_to_convergence,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def test_cosmology():
    """Standard test cosmology."""
    return Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )


@pytest.fixture
def test_density_field():
    """Create a simple test density field (1 + delta)."""
    # Small test field: 16x16x16
    mesh_shape = (16, 16, 16)

    # Create a simple density fluctuation pattern
    key = jr.key(42)
    delta = 0.1 * jr.normal(key, mesh_shape)  # Small fluctuations

    return 1.0 + delta  # 1 + delta


def test_convergence_Born_shape(test_cosmology, test_density_field):
    """Test that density_field_to_convergence returns correct shape."""
    box_shape = (100.0, 100.0, 100.0)  # Mpc/h
    field_size_deg = 2.0
    npix = 32
    z_source = 1100.0

    kappa = density_field_to_convergence(
        test_density_field,
        box_shape,
        test_cosmology,
        field_size_deg,
        npix,
        z_source,
    )

    # Single source (scalar) should give 2D output [npix, npix]
    assert kappa.ndim == 2, f"Expected 2D output for scalar z_source, got {kappa.ndim}D"
    assert kappa.shape == (npix, npix), f"Expected ({npix}, {npix}) shape, got {kappa.shape}"


def test_convergence_Born_values(test_cosmology, test_density_field):
    """Test that convergence values are reasonable."""
    box_shape = (100.0, 100.0, 100.0)
    field_size_deg = 2.0
    npix = 32
    z_source = 1100.0

    kappa = density_field_to_convergence(
        test_density_field,
        box_shape,
        test_cosmology,
        field_size_deg,
        npix,
        z_source,
    )

    # Convergence should be small for small density fluctuations
    assert jnp.all(jnp.isfinite(kappa)), "Convergence contains NaN or Inf"
    assert jnp.abs(jnp.mean(kappa)) < 1.0, f"Mean convergence too large: {jnp.mean(kappa)}"
    assert jnp.std(kappa) < 1.0, f"Convergence std too large: {jnp.std(kappa)}"


def test_convergence_Born_zero_field(test_cosmology):
    """Test that uniform density (no fluctuations) gives zero convergence."""
    mesh_shape = (16, 16, 16)
    density_field = jnp.ones(mesh_shape)  # Uniform density, no fluctuations

    box_shape = (100.0, 100.0, 100.0)
    field_size_deg = 2.0
    npix = 32
    z_source = 1100.0

    kappa = density_field_to_convergence(
        density_field,
        box_shape,
        test_cosmology,
        field_size_deg,
        npix,
        z_source,
    )

    # Should be approximately zero (within numerical precision)
    assert jnp.allclose(kappa, 0.0, atol=1e-6), f"Expected ~0 for uniform density, got {jnp.max(jnp.abs(kappa))}"


def test_density_field_to_convergence_single_source(test_cosmology, test_density_field):
    """Test density_field_to_convergence with single z_source."""
    box_shape = (100.0, 100.0, 100.0)  # Mpc/h
    field_size_deg = 2.0
    npix = 32
    z_source = 1100.0  # Single scalar value

    kappa = density_field_to_convergence(
        test_density_field,
        box_shape,
        test_cosmology,
        field_size_deg,
        npix,
        z_source,
    )

    # Scalar z_source returns 2D array [npix, npix]
    assert kappa.ndim == 2, f"Expected 2D output for scalar z_source, got {kappa.ndim}D"
    assert kappa.shape == (npix, npix), f"Expected shape ({npix}, {npix}), got {kappa.shape}"


def test_density_field_to_convergence_multiple_sources(test_cosmology, test_density_field):
    """Test density_field_to_convergence with multiple z_sources."""
    box_shape = (100.0, 100.0, 100.0)
    field_size_deg = 2.0
    npix = 32
    z_sources = jnp.array([0.5, 1.0, 1100.0])  # Multiple sources

    kappa = density_field_to_convergence(
        test_density_field,
        box_shape,
        test_cosmology,
        field_size_deg,
        npix,
        z_sources,
    )

    # Should return 3D array [n_sources, npix, npix]
    assert kappa.ndim == 3, f"Expected 3D output for multiple sources, got {kappa.ndim}D"
    assert kappa.shape == (len(z_sources), npix, npix), f"Expected shape {(len(z_sources), npix, npix)}, got {kappa.shape}"


def test_convergence_scales_with_amplitude(test_cosmology):
    """Test that convergence scales linearly with density amplitude (Born approximation)."""
    mesh_shape = (16, 16, 16)
    box_shape = (100.0, 100.0, 100.0)
    field_size_deg = 2.0
    npix = 32
    z_source = 1100.0

    # Create density field with known amplitude
    key = jr.key(42)
    delta = jr.normal(key, mesh_shape)

    # Test two amplitudes
    amp1, amp2 = 0.1, 0.2
    density_field1 = 1.0 + amp1 * delta
    density_field2 = 1.0 + amp2 * delta

    kappa1 = density_field_to_convergence(density_field1, box_shape, test_cosmology, field_size_deg, npix, z_source)
    kappa2 = density_field_to_convergence(density_field2, box_shape, test_cosmology, field_size_deg, npix, z_source)

    # In Born approximation, convergence should scale linearly
    ratio = jnp.mean(jnp.abs(kappa2)) / jnp.mean(jnp.abs(kappa1))
    expected_ratio = amp2 / amp1

    assert jnp.allclose(ratio, expected_ratio, rtol=0.1), (
        f"Convergence should scale linearly: expected ratio {expected_ratio:.2f}, got {ratio:.2f}"
    )


def test_convergence_different_field_sizes(test_cosmology, test_density_field):
    """Test that different field sizes produce consistent results."""
    box_shape = (100.0, 100.0, 100.0)
    npix = 32
    z_source = 1100.0

    # Test different angular field sizes
    field_sizes = [1.0, 2.0, 5.0]
    kappas = []

    for field_size in field_sizes:
        kappa = density_field_to_convergence(
            test_density_field,
            box_shape,
            test_cosmology,
            field_size,
            npix,
            z_source,
        )
        kappas.append(kappa)

    # All should produce finite results
    for i, kappa in enumerate(kappas):
        assert jnp.all(jnp.isfinite(kappa)), f"Field size {field_sizes[i]}° produced NaN/Inf"


def test_convergence_gradient(test_cosmology, test_density_field):
    """Test that convergence is differentiable (for inference)."""
    box_shape = (100.0, 100.0, 100.0)
    field_size_deg = 2.0
    npix = 16  # Smaller for gradient test
    z_source = 1100.0

    # Compute gradient w.r.t. density field
    grad_fn = jax.grad(lambda d: jnp.sum(density_field_to_convergence(d, box_shape, test_cosmology, field_size_deg, npix, z_source) ** 2))

    # Should compute without error
    grad = grad_fn(test_density_field)

    assert grad.shape == test_density_field.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN or Inf"


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
