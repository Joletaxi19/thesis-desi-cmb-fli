# Data access plan

No survey products are distributed with this repository. As the analysis is
implemented, acquisition steps should be documented for each dataset:

- **DESI galaxies** – tracer selections, release versions, and any preprocessing
  scripts (masking, depth cuts, random catalogs).
- **Planck PR4 CMB lensing** – download endpoints, filtering steps, and
  conversion to the internal map format.
- **ACT DR6+ CMB lensing** – access requirements and calibration steps required
  for cross-correlation with DESI.

All data handling occurs on NERSC systems rather than in a local `data/`
directory. Populate the following shared paths (or adapt `configs/nersc/perlmutter.yml`
if your allocation uses different locations):

```
/global/cfs/cdirs/desi/users/<username>/desi-cmb-fli/inputs/    # Read-only DESI catalogs, masks
/pscratch/sd/<u>/<username>/desi-cmb-fli/interim/               # Cached development products
/pscratch/sd/<u>/<username>/desi-cmb-fli/processed/             # Analysis-ready summaries and chains
```

The `<u>` placeholder is the first character of your NERSC username, matching
NERSC's scratch path convention.

Document the provenance of each file (checksums, licences, access controls)
alongside the instructions you record here so that collaborators can reproduce
your environment on Perlmutter.
