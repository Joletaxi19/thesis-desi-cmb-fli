# Data placeholders

No external datasets ship with this repository. Create your own data layout once
you begin implementing the analysis. A common pattern is:

```
data/
|- raw/       # Original survey/catalogue files (not tracked in git)
|- interim/   # Temporary or cached products
`- processed/ # Final data products used in results
```

Document any download scripts, checksums, or storage requirements here so the
future project remains reproducible.
