"""Ensure the stripped package still exposes basic metadata."""


def test_imports():
    import desi_cmb_fli

    assert hasattr(desi_cmb_fli, "__version__")
    assert isinstance(desi_cmb_fli.__version__, str)
