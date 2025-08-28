import numpy as np
import pytest
import healpy as hp
import fitsio

from desi_cmb_fli.data.loaders import load_planck_kappa_pr4, load_desi_lrg_catalog


def test_load_planck_kappa_pr4(tmp_path):
    nside = 2
    kappa = np.arange(hp.nside2npix(nside), dtype=np.float64)
    path = tmp_path / "kappa.fits"
    hp.write_map(path, kappa, dtype=np.float64, overwrite=True)
    data = load_planck_kappa_pr4(path)
    assert data["nside"] == nside
    assert np.allclose(data["kappa"], kappa)


def test_load_planck_kappa_pr4_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_planck_kappa_pr4(tmp_path / "missing.fits")


def test_load_desi_lrg_catalog(tmp_path):
    path = tmp_path / "lrg.fits"
    ra = np.array([0.1, 1.2])
    dec = np.array([-0.5, 2.5])
    z = np.array([0.5, 1.0])
    with fitsio.FITS(path, "rw", clobber=True) as f:
        f.write([ra, dec, z], names=["RA", "DEC", "Z"])
    data = load_desi_lrg_catalog(path)
    assert np.allclose(data["ra"], ra)
    assert np.allclose(data["dec"], dec)
    assert np.allclose(data["z"], z)


def test_load_desi_lrg_catalog_missing(tmp_path):
    with pytest.raises(OSError):
        load_desi_lrg_catalog(tmp_path / "missing.fits")
