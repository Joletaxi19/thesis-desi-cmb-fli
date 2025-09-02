"""Tests for power spectrum analysis utilities."""

import numpy as np

from desi_cmb_fli.analysis.power import bin_power, power_2d, radial_binning, spectra_3x2pt


def test_power_2d_auto():
    """Auto-power should be real and positive."""
    np.random.seed(42)
    field = np.random.normal(size=(32, 32))
    power = power_2d(field)
    assert power.shape == (32, 32)
    # Auto-power should be real at k=0
    assert np.isreal(power[0, 0])
    # For auto-power, the result should be real (within numerical precision)
    # since P_aa(k) = |FFT(a)|^2 = FFT(a) * conj(FFT(a))
    assert np.allclose(np.imag(power), 0, atol=1e-10)


def test_power_2d_cross():
    """Cross-power between different fields."""
    np.random.seed(42)
    field1 = np.random.normal(size=(16, 16))
    field2 = np.random.normal(size=(16, 16))
    power = power_2d(field1, field2)
    assert power.shape == (16, 16)
    # Cross-power is generally complex
    assert power.dtype == np.complex128


def test_radial_binning():
    """Radial binning should create reasonable bins."""
    kx = np.linspace(-1, 1, 8)
    ky = np.linspace(-1, 1, 8)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2)

    binning = radial_binning(kk, nbin=3)
    assert len(binning.edges) == 4  # nbin + 1
    assert len(binning.centers) == 3
    assert len(binning.indices) == 3

    # Check that bins don't include the DC mode
    flat_kk = kk.ravel()
    dc_idx = np.where(flat_kk == 0.0)[0]
    for bin_indices in binning.indices:
        assert not np.any(np.isin(bin_indices, dc_idx))


def test_bin_power():
    """Binning should preserve mean properties."""
    np.random.seed(42)
    # Create a simple 2D power array
    power = np.random.exponential(size=(16, 16)) + 1j * np.random.normal(size=(16, 16))
    power[0, 0] = 5.0  # Set DC mode

    kx = np.linspace(-1, 1, 16)
    ky = np.linspace(-1, 1, 16)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2)

    binning = radial_binning(kk, nbin=4)
    mean, var = bin_power(power, binning)

    assert len(mean) == 4
    assert len(var) == 4
    assert all(np.isfinite(mean))
    assert all(var >= 0)  # Variances should be non-negative


def test_spectra_3x2pt():
    """3x2pt spectra should return expected structure."""
    np.random.seed(42)
    g = np.random.normal(size=(32, 32))
    gamma_e = np.random.normal(size=(32, 32))

    kx = np.linspace(-1, 1, 32)
    ky = np.linspace(-1, 1, 32)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2)

    spectra = spectra_3x2pt(g, gamma_e, kk, nbin=5)

    required_keys = ["Pgg", "PgE", "PEE"]
    assert all(key in spectra for key in required_keys)

    for key in required_keys:
        assert "k" in spectra[key]
        assert "mean" in spectra[key]
        assert "var" in spectra[key]
        assert len(spectra[key]["k"]) == 5
        assert len(spectra[key]["mean"]) == 5
        assert len(spectra[key]["var"]) == 5
        # Auto-spectra should be real
        if key in ["Pgg", "PEE"]:
            assert np.all(np.isreal(spectra[key]["mean"]))


def test_empty_bins():
    """Empty bins should be handled gracefully."""
    # Create a case where some bins will be empty
    power = np.ones((8, 8), dtype=complex)
    kx = np.linspace(0, 2, 8)
    ky = np.linspace(0, 2, 8)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2)

    # Use many bins to ensure some are empty
    binning = radial_binning(kk, nbin=20)
    mean, var = bin_power(power, binning)

    # Check that empty bins return sensible values
    empty_bins = [i for i, idx in enumerate(binning.indices) if len(idx) == 0]
    single_bins = [i for i, idx in enumerate(binning.indices) if len(idx) == 1]

    if empty_bins:  # If there are empty bins
        for i in empty_bins:
            assert mean[i] == 0.0
            assert var[i] == np.inf

    if single_bins:  # If there are single-element bins
        for i in single_bins:
            assert var[i] == 0.0  # No variance estimate possible for single elements
