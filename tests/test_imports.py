"""Ensure the top-level package imports cleanly and exposes metadata."""


def test_imports():
    import desi_cmb_fli

    assert hasattr(desi_cmb_fli, "__version__")
