# Data layout

The repository does not store heavy data. Create the following directories locally:

```
data/
├─ raw/         # Inputs (DESI catalogs, Planck PR4 kappa maps, etc.)
├─ interim/     # Temporary products
└─ processed/   # Final derived products (masked maps, spectra, etc.)
```

See `docs/data_access.md` for instructions to obtain DESI and Planck PR4 datasets.
