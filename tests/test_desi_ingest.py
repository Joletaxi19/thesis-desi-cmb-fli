"""Tests for DESI data ingestion (NERSC only)."""
import os

import pytest

# Skip all tests in this file if not on NERSC
pytestmark = pytest.mark.skipif(
    not os.path.exists("/global/cfs/cdirs/desi"),
    reason="DESI data only available on NERSC",
)


def test_desi_lss_catalog_structure():
    """Test that DESI DR1 LSS clustering catalogs can be read."""
    import fitsio

    base_dir = os.environ.get(
        "DESI_LSS_CLUS_DIR",
        "/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5",
    )

    # Test with LRG NGC as example
    data_path = f"{base_dir}/LRG_NGC_clustering.dat.fits"
    rand_path = f"{base_dir}/LRG_NGC_0_clustering.ran.fits"

    # Read catalogs
    data = fitsio.read(data_path, ext=1)
    rand = fitsio.read(rand_path, ext=1)

    # Basic checks
    assert len(data) > 0, "Data catalog is empty"
    assert len(rand) > 0, "Random catalog is empty"

    # Check required columns
    for arr, name in [(data, "data"), (rand, "rand")]:
        cols = set(arr.dtype.names or [])
        for req in ("RA", "DEC", "Z"):
            assert req in cols, f"Missing {req} in {name} catalog"

    # Check z-range is reasonable
    assert 0 < data["Z"].min() < 3, "Data redshifts out of range"
    assert 0 < data["Z"].max() < 3, "Data redshifts out of range"


def test_desi_catalog_with_z_cut():
    """Test filtering DESI catalogs by redshift."""
    import fitsio

    base_dir = os.environ.get(
        "DESI_LSS_CLUS_DIR",
        "/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5",
    )

    data_path = f"{base_dir}/LRG_NGC_clustering.dat.fits"
    data = fitsio.read(data_path, ext=1)

    # Apply z-cut
    zmin, zmax = 0.6, 0.8
    mask = (data["Z"] >= zmin) & (data["Z"] < zmax)
    filtered = data[mask]

    assert len(filtered) > 0, "No data in z-range"
    assert filtered["Z"].min() >= zmin, "Z-cut failed (min)"
    assert filtered["Z"].max() < zmax, "Z-cut failed (max)"
