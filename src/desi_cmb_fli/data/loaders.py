from __future__ import annotations

from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np


def load_planck_kappa_pr4(path: str | Path) -> dict[str, Any]:
    """Load Planck PR4 kappa map (FITS). Minimal stub."""
    m = hp.read_map(path, field=0, dtype=np.float64)
    nside = hp.get_nside(m)
    return {"kappa": m, "nside": nside}


def load_desi_lrg_catalog(path: str | Path) -> dict[str, Any]:
    """Load a minimal subset from a DESI LRG catalog (stub)."""
    import fitsio

    with fitsio.FITS(str(path)) as f:
        data = f[1].read(columns=["RA", "DEC", "Z"])
    return {"ra": data["RA"], "dec": data["DEC"], "z": data["Z"]}
