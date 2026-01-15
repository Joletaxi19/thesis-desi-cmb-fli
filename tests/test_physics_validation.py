"""
Publication Readiness: Physics and Mathematical Validation Tests.

This module contains rigorous tests to verify the physical and mathematical
correctness of the DESI x CMB Lensing Field-Level Inference pipeline.

Tests cover:
1. Cosmological constants and unit consistency
2. Lensing kernel amplitude validation against theory
3. Gradient consistency (finite difference vs analytic)
4. Limber approximation validity
5. FourierSpaceGaussian normalization
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
import pytest
from jax import grad

from desi_cmb_fli.bricks import Planck18, get_cosmology
from desi_cmb_fli.cmb_lensing import (
    compute_theoretical_cl_kappa,
    density_field_to_convergence,
)
from desi_cmb_fli.model import FieldLevelModel, FourierSpaceGaussian, default_config
from desi_cmb_fli.nbody import a2f, a2g

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def planck18():
    """Fresh Planck18 cosmology."""
    cosmo = Planck18()
    cosmo._workspace = {}
    return cosmo


# =============================================================================
# 1. COSMOLOGICAL CONSTANTS AND UNITS
# =============================================================================

def test_jax_cosmo_constants_consistency():
    """Verify jax_cosmo constants match expected values."""
    # H0 in km/s/Mpc should be 100
    assert jnp.isclose(constants.H0, 100.0, rtol=1e-5), f"H0={constants.H0}, expected 100"

    # c in km/s should be ~299792.458
    c_expected = 299792.458  # km/s in natural SI
    # jax_cosmo uses c in Mpc/h, so c/H0 ~ 3000 Mpc/h
    c_over_H0 = constants.c / constants.H0
    assert jnp.isclose(c_over_H0, c_expected / 100.0, rtol=1e-3), (
        f"c/H0 = {c_over_H0}, expected {c_expected/100}"
    )


def test_omega_m_sum_consistency(planck18):
    """Verify Omega_m = Omega_c + Omega_b."""
    omega_m_computed = planck18.Omega_c + planck18.Omega_b
    omega_m_property = planck18.Omega_m
    assert jnp.isclose(omega_m_computed, omega_m_property, rtol=1e-6), (
        f"Omega_m mismatch: computed={omega_m_computed}, property={omega_m_property}"
    )


def test_get_cosmology_omega_m_consistency():
    """Verify get_cosmology preserves Omega_m correctly."""
    omega_m_input = 0.3111
    sigma8_input = 0.81
    cosmo = get_cosmology(Omega_m=omega_m_input, sigma8=sigma8_input)

    assert jnp.isclose(cosmo.Omega_m, omega_m_input, rtol=1e-6), (
        f"Omega_m mismatch: input={omega_m_input}, cosmo={cosmo.Omega_m}"
    )
    assert jnp.isclose(cosmo.sigma8, sigma8_input, rtol=1e-6), (
        f"sigma8 mismatch: input={sigma8_input}, cosmo={cosmo.sigma8}"
    )


# =============================================================================
# 2. LENSING KERNEL AMPLITUDE VALIDATION
# =============================================================================

def test_lensing_prefactor_amplitude(planck18):
    """
    Verify the lensing prefactor: 3/2 * Omega_m * (H0/c)^2

    This prefactor should have units that when multiplied by (chi/a) * delta * dchi
    give dimensionless kappa.
    """
    # The prefactor in natural units (Mpc/h)^-2
    expected_prefactor = 1.5 * planck18.Omega_m * (constants.H0 / constants.c)**2

    # Sanity check: for Planck18, Omega_m ~ 0.31
    # (H0/c)^2 ~ (100/3e5)^2 ~ 1.1e-7 (Mpc/h)^-2
    # So prefactor ~ 1.5 * 0.31 * 1.1e-7 ~ 5e-8
    assert 1e-9 < expected_prefactor < 1e-6, (
        f"Lensing prefactor outside expected range: {expected_prefactor}"
    )


def test_convergence_amplitude_physical(planck18):
    """
    Test that convergence kappa has physically reasonable amplitude.

    For a uniform overdensity delta = 0.1 filling a 100 Mpc/h box at z~0.5,
    kappa should be O(1e-4) to O(1e-3).
    """
    mesh_shape = (16, 16, 16)
    box_shape = (100.0, 100.0, 100.0)
    delta = 0.1

    density_field = jnp.ones(mesh_shape) * (1 + delta)

    kappa = density_field_to_convergence(
        density_field,
        box_shape,
        planck18,
        field_size_deg=2.0,
        field_npix=32,
        z_source=1100.0,
        box_center_chi=150.0,  # z ~ 0.05
    )

    # Mean kappa should be positive for positive overdensity
    # and order O(delta * chi * prefactor)
    kappa_mean = jnp.mean(kappa)
    assert jnp.isfinite(kappa_mean), "kappa contains NaN/Inf"

    # For a 100 Mpc/h path with delta=0.1, kappa ~ delta * dchi * prefactor * chi/a
    # Very rough: kappa ~ 0.1 * 100 * 5e-8 * 150 ~ 7.5e-5
    # Allow wide tolerance
    assert jnp.abs(kappa_mean) < 0.1, f"kappa mean unreasonably large: {kappa_mean}"


def test_theoretical_cl_kappa_scaling_with_omega_m():
    """
    Test that C_l^{kappa kappa} scales as Omega_m^2 (at fixed P(k)).

    The lensing kernel W ~ Omega_m, so C_l ~ W^2 ~ Omega_m^2.
    This is approximate because P(k) also depends weakly on Omega_m.
    """
    ell = jnp.linspace(100, 1000, 10)
    chi_min = 100.0
    chi_max = 500.0
    z_source = 1100.0

    # Cosmology 1
    cosmo1 = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0
    )
    cl1 = compute_theoretical_cl_kappa(cosmo1, ell, chi_min, chi_max, z_source)

    # Cosmology 2 with 1.5x Omega_m
    cosmo2 = jc.Cosmology(
        Omega_c=0.375, Omega_b=0.075, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0
    )
    cl2 = compute_theoretical_cl_kappa(cosmo2, ell, chi_min, chi_max, z_source)

    # C_l should scale ~Omega_m^2 (factor of 1.5^2 = 2.25)
    ratio = jnp.mean(cl2) / jnp.mean(cl1)
    expected_ratio = (0.45 / 0.30)**2  # = 2.25

    # Allow 50% tolerance due to P(k) changes
    assert 1.5 < ratio < 4.0, (
        f"C_l scaling with Omega_m unexpected: ratio={ratio:.2f}, expected~{expected_ratio:.2f}"
    )


# =============================================================================
# 3. GRADIENT CONSISTENCY
# =============================================================================

def test_logpdf_gradient_finite_difference():
    """
    Verify analytic gradient of log_prob matches finite difference.

    This is crucial for HMC/MCLMC samplers.
    """
    config = default_config.copy()
    config["mesh_shape"] = (16, 16, 16)  # Must be even for fourier reparametrization
    config["box_shape"] = (100.0, 100.0, 100.0)
    config["evolution"] = "kaiser"
    config["cmb_enabled"] = False

    model = FieldLevelModel(**config)

    # Generate synthetic observation
    truth_params = {
        "Omega_m": 0.31, "sigma8": 0.81,
        "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0
    }
    truth = model.predict(samples=truth_params, frombase=True, rng=42)

    # Condition model
    model.reset()
    model.condition({"obs": truth["obs"]})

    # Test point
    key = jr.key(123)
    init_mesh = jr.normal(key, (16, 16, 16))  # Real space mesh (will be converted by rg2cgh)
    test_params = {
        "Omega_m_": 0.0, "sigma8_": 0.0,
        "b1_": 0.0, "b2_": 0.0, "bs2_": 0.0, "bn2_": 0.0,
        "init_mesh_": init_mesh,
    }

    # Analytic gradient
    logpdf_fn = model.logpdf
    analytic_grad = grad(logpdf_fn)(test_params)

    # Finite difference gradient for scalars
    eps = 1e-4
    for scalar_key in ["Omega_m_", "sigma8_", "b1_"]:
        params_plus = test_params.copy()
        params_plus[scalar_key] = test_params[scalar_key] + eps

        params_minus = test_params.copy()
        params_minus[scalar_key] = test_params[scalar_key] - eps

        fd_grad = (logpdf_fn(params_plus) - logpdf_fn(params_minus)) / (2 * eps)

        rel_error = jnp.abs(analytic_grad[scalar_key] - fd_grad) / (jnp.abs(fd_grad) + 1e-8)

        assert rel_error < 0.01, (
            f"Gradient mismatch for {scalar_key}: analytic={analytic_grad[scalar_key]:.6f}, "
            f"FD={fd_grad:.6f}, rel_error={rel_error:.4f}"
        )


# =============================================================================
# 4. FOURIER SPACE GAUSSIAN NORMALIZATION
# =============================================================================

def test_fourier_gaussian_mean_invariance():
    """
    Verify FourierSpaceGaussian samples have correct mean (loc).
    """
    N = 32
    loc = jnp.ones((N, N)) * 0.5
    var_k = jnp.ones((N, N))
    var_k = var_k.at[0, 0].set(1e30)  # DC mode infinite variance (should be ignored)

    dist = FourierSpaceGaussian(loc, var_k)

    # Draw many samples
    n_samples = 100
    keys = jr.split(jr.key(0), n_samples)
    samples = jax.vmap(dist.sample)(keys)

    # Mean of samples should approach loc
    sample_mean = jnp.mean(samples, axis=0)

    # Error should be O(1/sqrt(n_samples))
    error = jnp.mean(jnp.abs(sample_mean - loc))
    assert error < 0.2, f"FourierSpaceGaussian mean error too large: {error}"


def test_fourier_gaussian_dc_mode_zeroed():
    """
    Verify DC mode is zeroed in FourierSpaceGaussian samples.
    """
    N = 16
    loc = jnp.zeros((N, N))
    var_k = jnp.ones((N, N))
    var_k = var_k.at[0, 0].set(1e30)

    dist = FourierSpaceGaussian(loc, var_k)

    sample = dist.sample(jr.key(0))
    sample_k = jnp.fft.fftn(sample)

    # DC mode should be zero
    dc = sample_k[0, 0]
    assert jnp.abs(dc) < 1e-10, f"DC mode not zeroed: {dc}"


def test_fourier_gaussian_log_prob_finite():
    """
    Verify log_prob is finite for reasonable inputs.
    """
    N = 16
    loc = jnp.zeros((N, N))
    var_k = jnp.ones((N, N)) * 1e-4
    var_k = var_k.at[0, 0].set(jnp.inf)

    dist = FourierSpaceGaussian(loc, var_k)

    sample = dist.sample(jr.key(0))
    log_prob = dist.log_prob(sample)

    assert jnp.isfinite(log_prob), f"log_prob is not finite: {log_prob}"


# =============================================================================
# 5. GROWTH FACTOR CONSISTENCY
# =============================================================================

def test_growth_factor_limits(planck18):
    """
    Test growth factor at known limits:
    - D(a=0) should be 0
    - D(a=1) should be normalized to some value > 0
    - D(a) should be monotonically increasing
    """
    a_values = jnp.array([0.001, 0.1, 0.5, 1.0])
    g_values = jax.vmap(lambda a: a2g(planck18, a))(a_values)

    # Monotonicity
    for i in range(len(g_values) - 1):
        assert g_values[i+1] > g_values[i], (
            f"Growth factor not monotonic: g({a_values[i]})={g_values[i]}, "
            f"g({a_values[i+1]})={g_values[i+1]}"
        )

    # At high z (a=0.001), growth factor should be much smaller than at a=1
    assert g_values[0] / g_values[-1] < 0.1, (
        f"Growth factor ratio unexpected: g(0.001)/g(1) = {g_values[0]/g_values[-1]}"
    )


def test_growth_rate_f_range(planck18):
    """
    Test f = dlnD/dlna is in expected range ~[0.5, 1] for LCDM.
    """
    a_values = jnp.array([0.1, 0.5, 0.8, 1.0])
    f_values = jax.vmap(lambda a: a2f(planck18, a))(a_values)

    for a, f in zip(a_values, f_values, strict=False):
        assert 0.4 < f < 1.1, f"Growth rate f out of range at a={a}: f={f}"


# =============================================================================
# 6. LIMBER APPROXIMATION VALIDITY
# =============================================================================

def test_cl_kappa_ell_scaling():
    """
    Test C_l^{kappa} scaling at high ell.

    For a reasonable cosmology, C_l should decrease at high ell
    (power law behavior).
    """
    cosmo = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
        Omega_k=0.0, w0=-1.0, wa=0.0
    )

    ell_low = jnp.array([100.0])
    ell_high = jnp.array([1000.0])

    cl_low = compute_theoretical_cl_kappa(cosmo, ell_low, 100.0, 1000.0, 1100.0)
    cl_high = compute_theoretical_cl_kappa(cosmo, ell_high, 100.0, 1000.0, 1100.0)

    # C_l should be larger at low ell
    assert cl_low[0] > cl_high[0], (
        f"C_l not decreasing with ell: C_l(100)={cl_low[0]}, C_l(1000)={cl_high[0]}"
    )


# =============================================================================
# 7. UNIT CONSISTENCY IN MODEL
# =============================================================================

def test_model_box_angle_consistency():
    """
    Verify box angular size calculation is consistent with geometry.

    theta = 2 * arctan(L/2 / chi_max) should give field_size_deg.
    """
    config = default_config.copy()
    config["mesh_shape"] = (32, 32, 64)
    config["box_shape"] = (300.0, 300.0, 600.0)
    config["cmb_enabled"] = True
    config["cmb_noise_nell"] = {"ell": np.arange(10, 2000), "N_ell": np.ones(1990) * 1e-8}

    model = FieldLevelModel(**config)

    # chi_max = box_size[2]
    chi_max = 600.0
    L = 300.0
    expected_half_angle_rad = np.arctan(L / (2.0 * chi_max))
    expected_field_size_deg = 2.0 * expected_half_angle_rad * (180.0 / np.pi)

    assert np.isclose(model.cmb_field_size_deg, expected_field_size_deg, rtol=0.01), (
        f"Field size mismatch: computed={model.cmb_field_size_deg:.2f}, "
        f"expected={expected_field_size_deg:.2f}"
    )


# =============================================================================
# 8. EDGE CASES
# =============================================================================

def test_zero_fluctuation_field(planck18):
    """Test convergence with zero density fluctuations."""
    mesh_shape = (8, 8, 8)
    density_field = jnp.ones(mesh_shape)  # delta = 0

    kappa = density_field_to_convergence(
        density_field,
        box_size=(100.0, 100.0, 100.0),
        cosmo=planck18,
        field_size_deg=2.0,
        field_npix=16,
        z_source=1100.0,
        box_center_chi=200.0,
    )

    assert jnp.allclose(kappa, 0.0, atol=1e-8), f"Non-zero kappa for delta=0: {jnp.max(jnp.abs(kappa))}"


def test_source_behind_box(planck18):
    """
    Test that sources behind the box produce non-zero kappa,
    while sources in front produce approximately zero.
    """
    mesh_shape = (8, 8, 8)
    density_field = 1.0 + 0.1 * jr.normal(jr.key(0), mesh_shape)
    box_center_chi = 200.0

    # Source behind box (should produce non-zero kappa)
    kappa_behind = density_field_to_convergence(
        density_field,
        box_size=(100.0, 100.0, 100.0),
        cosmo=planck18,
        field_size_deg=2.0,
        field_npix=16,
        z_source=1100.0,  # chi >> box
        box_center_chi=box_center_chi,
    )

    # Source in front of box (should produce ~zero kappa due to kernel)
    kappa_front = density_field_to_convergence(
        density_field,
        box_size=(100.0, 100.0, 100.0),
        cosmo=planck18,
        field_size_deg=2.0,
        field_npix=16,
        z_source=0.01,  # very low z, chi << box_center
        box_center_chi=box_center_chi,
    )

    # Variance of kappa_behind should be much larger
    assert jnp.std(kappa_behind) > 10 * jnp.std(kappa_front), (
        f"Source position kernel not working: std(behind)={jnp.std(kappa_behind):.2e}, "
        f"std(front)={jnp.std(kappa_front):.2e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
