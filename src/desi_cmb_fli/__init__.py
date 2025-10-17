"""Top-level package metadata exposed by desi_cmb_fli."""

from importlib.metadata import PackageNotFoundError, version

try:
	__version__ = version("desi-cmb-fli")
except PackageNotFoundError:
	__version__ = "0.1.0"

__all__ = ["__version__"]
