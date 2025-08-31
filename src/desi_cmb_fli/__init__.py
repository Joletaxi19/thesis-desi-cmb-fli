"""Top-level package for desi_cmb_fli.

Exposes a small, predictable namespace and a robust ``__version__`` attribute
that works both when the package is installed and when it is imported from a
source checkout (e.g., during development or CI).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__all__ = ["data", "sim", "inference", "analysis", "utils"]

try:
    # Normal case: resolve installed package version from importlib.metadata.
    __version__ = _version("desi-cmb-fli")
except PackageNotFoundError:  # Fallback when package is not installed
    # During development ("editable" or direct source imports) there may be no
    # installed distribution metadata available.
    __version__ = "0.0.0"
