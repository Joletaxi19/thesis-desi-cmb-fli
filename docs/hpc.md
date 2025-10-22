# NERSC & Perlmutter quickstart

This note collects the minimal information needed to run the thesis workflow on
NERSC, with a focus on Perlmutter GPU jobs. Update the placeholders with your
project details before launching.

## Accounts and projects
- Ensure your NERSC account is active and associated with the DESI allocation.
- Confirm access to the `cosmo` software stack and shared project spaces under
  `/global/cfs/cdirs/`.
- Keep your project ID handy; the example configs assume a placeholder `m####`.

## Environment bootstrap
1. Log in to Perlmutter via `ssh username@perlmutter.nersc.gov`.
2. Load the CUDA-enabled Python stack provided by NERSC:
   ```bash
   module load python
   module load cudatoolkit
   source "$(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh"
   ```
3. Create the thesis environment inside `$SCRATCH/envs` (or another writable
  location) using the repository recipe:
  ```bash
  mkdir -p ${SCRATCH}/envs
  conda env create --solver classic -p ${SCRATCH}/envs/desi-cmb-fli -f env/environment.yml
  conda activate ${SCRATCH}/envs/desi-cmb-fli
  pip install -e ".[dev,analysis]"
  ```
4. Replace CPU JAX wheels with the GPU build once inside the login node:
   ```bash
   pip install --upgrade "jax[cuda12_pip]" "jaxlib[cuda12_pip]" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```
   Adjust the CUDA version if NERSC updates the default toolchain.

## Shared storage layout
- **Permanent project space**: `/global/cfs/cdirs/desi/users/<username>/desi-cmb-fli`
  for curated inputs, configs, and checkpoints shared across jobs.
- **Scratch (recommended for large outputs)**: `/pscratch/sd/<u>/<username>/desi-cmb-fli`
  for transient products such as sampler chains or mock realizations.
- The repository assumes those absolute paths instead of a local `data/`
  directory. Update `configs/nersc/perlmutter.yml` if your team maintains a
  different convention.
  Replace `<u>` with the first letter of your username.

## Batch submission
- Populate `configs/nersc/perlmutter.yml` with your NERSC username, project ID,
  and preferred queue.
- Use the template `configs/nersc/slurm/nuts_perlmutter.sbatch` as a starting
  point for launching NumPyro/NUTS chains. The script reads the YAML config at
  runtime to locate data, scratch areas, and environment activation commands.
- Submit jobs via `sbatch configs/nersc/slurm/nuts_perlmutter.sbatch`. Monitor
  with `squeue -u <username>`.

## Data integrity and transfers
- Pull DESI catalogs and masks directly from curated project directories on
  NERSC; document the exact source in `docs/data_access.md`.
- For small development subsets, consider copying to `/pscratch/.../cache/` and
  referencing that location in the appropriate config file.
- Use `hsi`/`htar` or Globus for any transfers that must leave NERSC.

Keep this document updated as NERSC modules or allocation details evolve. Short
notes about module versions or queue changes belong here to prevent divergence
between the documentation and operational reality.
