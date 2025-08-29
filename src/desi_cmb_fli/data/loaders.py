from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_planck_kappa_pr4(path: str | Path) -> dict[str, Any]:
    """Load a Planck PR4 kappa map from a FITS file.

    Parameters
    ----------
    path
        Path to a HEALPix FITS file containing the convergence (``kappa``)
        map in the first HDU.

    Returns
    -------
    dict
        Dictionary with the map under ``"kappa"`` and the associated
        ``"nside"`` value.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    ImportError
        If :mod:`healpy` is not installed.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Planck kappa map file not found: {path}")

    try:  # pragma: no cover - import guarded for informative error
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - informative error
        raise ImportError(
            "The 'healpy' package is required to load Planck PR4 kappa maps. "
            "Install it with 'pip install healpy'."
        ) from exc

    m = hp.read_map(path, field=0, dtype=np.float64)
    nside = hp.get_nside(m)
    return {"kappa": m, "nside": nside}


def load_desi_lrg_catalog(path: str | Path) -> dict[str, Any]:
    """Load a minimal subset from a DESI LRG catalog FITS file.

    Parameters
    ----------
    path
        Path to a FITS table containing ``RA``, ``DEC`` and ``Z`` columns.

    Returns
    -------
    dict
        Dictionary with arrays ``"ra"``, ``"dec"`` and ``"z"`` extracted
        from the corresponding columns.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    ImportError
        If :mod:`fitsio` is not installed.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DESI LRG catalog file not found: {path}")

    try:  # pragma: no cover - import guarded for informative error
        import fitsio
    except ImportError as exc:  # pragma: no cover - informative error
        raise ImportError(
            "The 'fitsio' package is required to load DESI LRG catalog "
            "files. Install it with 'pip install fitsio'."
        ) from exc

    with fitsio.FITS(str(path)) as f:
        data = f[1].read(columns=["RA", "DEC", "Z"])
    return {"ra": data["RA"], "dec": data["DEC"], "z": data["Z"]}
