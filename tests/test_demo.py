"""Smoke test for the toy FLI wiring."""

from desi_cmb_fli.inference.fli import toy_fli


def test_toy_fli_runs():
    """Pipeline runs and returns the expected summary structure."""
    out = toy_fli(seed=0)
    assert "summary" in out
    assert "toy_power" in out["summary"]
