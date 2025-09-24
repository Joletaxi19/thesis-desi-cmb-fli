# Data access plan

No survey products are distributed with this repository. As the analysis is
implemented, acquisition steps should be documented for each dataset:

- **DESI galaxies** – tracer selections, release versions, and any preprocessing
  scripts (masking, depth cuts, random catalogs).
- **Planck PR4 CMB lensing** – download endpoints, filtering steps, and
  conversion to the internal map format.
- **ACT DR6+ CMB lensing** – access requirements and calibration steps required
  for cross-correlation with DESI.

A recommended local layout is:

```
data/
|- raw/       # Read-only copies of official DESI and CMB lensing products
|- interim/   # Cached or down-sampled intermediates for development work
`- processed/ # Analysis-ready maps, masks, and summary statistics
```

For reproducibility, checksums, licences, and any proprietary access
constraints should accompany the acquisition instructions.
