from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__all__ = ["data", "sim", "inference", "analysis", "utils"]

try:
    __version__ = _version("desi-cmb-fli")
except PackageNotFoundError:  # Fallback when package is not installed
    __version__ = "0.0.0"
