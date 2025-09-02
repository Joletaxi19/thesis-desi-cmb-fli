"""Additional tests for configuration validation."""

import pytest

from desi_cmb_fli.sim.flat_sky import SynthConfig


def test_synthconfig_validation():
    """Configuration validation should catch invalid parameters."""
    # Valid configuration should work
    cfg = SynthConfig()
    assert cfg.nx == 128

    # Invalid grid dimensions
    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        SynthConfig(nx=0)

    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        SynthConfig(ny=-1)

    # Invalid box size
    with pytest.raises(ValueError, match="Box size must be positive"):
        SynthConfig(box_size=0)

    with pytest.raises(ValueError, match="Box size must be positive"):
        SynthConfig(box_size=-10)

    # Invalid amplitude
    with pytest.raises(ValueError, match="Amplitude A must be non-negative"):
        SynthConfig(A=-0.5)

    # Invalid noise levels
    with pytest.raises(ValueError, match="Noise levels must be non-negative"):
        SynthConfig(sigma_g=-0.1)

    with pytest.raises(ValueError, match="Noise levels must be non-negative"):
        SynthConfig(sigma_gamma=-0.2)

    # Invalid seed
    with pytest.raises(ValueError, match="Random seed must be non-negative"):
        SynthConfig(seed=-1)


def test_synthconfig_edge_cases():
    """Test edge cases for configuration."""
    # Zero noise should be allowed
    cfg = SynthConfig(sigma_g=0, sigma_gamma=0)
    assert cfg.sigma_g == 0
    assert cfg.sigma_gamma == 0

    # Zero amplitude should be allowed
    cfg = SynthConfig(A=0)
    assert cfg.A == 0

    # Very small grids should work
    cfg = SynthConfig(nx=2, ny=2)
    assert cfg.nx == 2
    assert cfg.ny == 2
