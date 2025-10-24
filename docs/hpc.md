# NERSC Perlmutter Setup

Quick guide for running on NERSC Perlmutter GPU nodes.

## Initial Setup (once)

```bash
# 1. Log in and load modules
ssh username@perlmutter.nersc.gov
module load python cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh

# 2. Create conda environment
mkdir -p ${SCRATCH}/envs
conda env create --solver libmamba -p ${SCRATCH}/envs/desi-cmb-fli -f env/environment.yml
conda activate ${SCRATCH}/envs/desi-cmb-fli
pip install -e .

# 3. Install JAX with GPU support
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Enable git hooks
pre-commit install
```

**Add to `~/.bashrc`** (required for GPU):
```bash
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}
```

## Daily Usage

```bash
module load python cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
```

**Test GPU** (requires GPU node):
```bash
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 1 --account=desi
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0)]
```

## Batch Jobs

Template script structure:
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=desi

module load python cudatoolkit/12.4
source /global/common/.../conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli

# Your code here
python your_script.py
```

Submit with: `sbatch script.sbatch`

## Storage

- **Project data**: `/global/cfs/cdirs/desi/users/<username>/desi-cmb-fli`
- **Scratch (large outputs)**: `/pscratch/sd/<u>/<username>/desi-cmb-fli`
- **DESI DR1 catalogs**: `/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5`

## Notes

- **jaxpm version**: Project uses `jaxpm==0.0.2` (stable). Newer versions (0.1.x+) have breaking API changes.
- **GPU on login nodes**: JAX falls back to CPU on login nodes (expected). GPU only works on compute nodes.
