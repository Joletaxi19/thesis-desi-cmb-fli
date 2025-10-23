#!/usr/bin/env python3
"""
Minimal, doc-compliant DESI DR1 LSS ingestion check.

Reads the official 'clustering-ready' catalogs directly with fitsio, using
the DESI data model naming convention:
  - Data   : {TARGET}_{PHOTSYS}_clustering.dat.fits
  - Randoms: {TARGET}_{PHOTSYS}_{RANN}_clustering.ran.fits  (RANN=0..17)

On NERSC (DR1), the directory is:
  /global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5
"""

import argparse
import os
import sys

import fitsio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default=os.environ.get("DESI_LSS_CLUS_DIR", ""),
        help="Directory with *clustering.dat.fits and *_clustering.ran.fits (e.g., DR1 v1.5).",
    )
    ap.add_argument("--tracer", required=True, help="Example: LRG, ELG, BGS_BRIGHT, QSO")
    ap.add_argument("--region", required=True, help="NGC or SGC (match file names)")
    ap.add_argument("--rand-index", type=int, default=0, help="Random index (0..17)")
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    args = ap.parse_args()

    if not args.base_dir:
        print("ERROR: --base-dir not provided and DESI_LSS_CLUS_DIR not set.", file=sys.stderr)
        sys.exit(2)

    base = args.base_dir.rstrip("/")
    data_path = f"{base}/{args.tracer}_{args.region}_clustering.dat.fits"
    rand_path = f"{base}/{args.tracer}_{args.region}_{args.rand_index}_clustering.ran.fits"

    # Read HDU 1 as per DESI data model
    data = fitsio.read(data_path, ext=1)
    rand = fitsio.read(rand_path, ext=1)

    print(f"[OK] data: {data_path}  (n={len(data)}, cols={list(data.dtype.names)[:8]}...)")
    print(f"[OK] rand: {rand_path}  (n={len(rand)}, cols={list(rand.dtype.names)[:8]}...)")

    # Minimal schema checks: RA, DEC, Z exist
    for arr, name in [(data, "data"), (rand, "rand")]:
        cols = set(arr.dtype.names or [])
        for req in ("RA", "DEC", "Z"):
            if req not in cols:
                raise RuntimeError(f"Missing required column in {name}: {req}")

    # Optional z filter + quick stats
    def _stats(tag, z):
        z = np.asarray(z)
        print(f"{tag}: z[min,max] = [{z.min():.3f}, {z.max():.3f}], n = {z.size}")

    if args.zmin is not None or args.zmax is not None:
        m_data = np.ones(len(data), dtype=bool)
        m_rand = np.ones(len(rand), dtype=bool)
        if args.zmin is not None:
            m_data &= data["Z"] >= args.zmin
            m_rand &= rand["Z"] >= args.zmin
        if args.zmax is not None:
            m_data &= data["Z"] < args.zmax
            m_rand &= rand["Z"] < args.zmax
        data = data[m_data]
        rand = rand[m_rand]
        print(f"[INFO] Applied z-filter: zmin={args.zmin}, zmax={args.zmax}")

    _stats("data", data["Z"])
    _stats("rand", rand["Z"])
    print("[DONE] Minimal ingestion and checks succeeded.")

if __name__ == "__main__":
    sys.exit(main())
