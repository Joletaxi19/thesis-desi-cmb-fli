"""Top-level package metadata exposed by desi_cmb_fli."""

from importlib.metadata import PackageNotFoundError, version

try:
	__version__ = version("desi-cmb-fli")
except PackageNotFoundError:
	__version__ = "0.1.0"

# Import main modules for convenience
from . import evolution, galaxy_bias, initial_conditions

__all__ = ["__version__", "initial_conditions", "evolution", "galaxy_bias"]
