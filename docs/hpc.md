# NERSC Perlmutter Setup

Quick guide for running on NERSC Perlmutter GPU nodes.

## Initial Setup (once)

```bash
# 1. Log in and source conda
ssh username@perlmutter.nersc.gov
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh

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

## Running Jobs

All scripts are configured via YAML files in `configs/<script_name>/config.yaml` which contain:
- SLURM parameters (time, nodes, GPUs, constraint)
- Model configuration
- MCMC settings
- Random seeds

### Submit a job

```bash
# Example : CMB lensing joint inference
./configs/inference/submit.py
```

The submit script reads the YAML configuration and submits the job with appropriate SLURM parameters.

### Check job status

```bash
squeue -u $USER
```

### View outputs

- Logs: `logs/<jobname>-<jobid>.out` (in home directory)
- Figures: `$SCRATCH/outputs/run_<timestamp>/figures/`
- Config used: `$SCRATCH/outputs/run_<timestamp>/config/`
- Samples: `$SCRATCH/outputs/run_<timestamp>/config/*.npz`

### Test GPU (interactive)

```bash
salloc --nodes 1 --qos interactive --time 00:03:00 --constraint gpu --gpus 1 --account=desi
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0), ...]
```

### Interactive job

```bash
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=desi
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20251214-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0), ...]

# Launch job directly
python scripts/run_inference.py --config configs/inference/config.yaml
```

## Storage

- **DESI DR1 catalogs**: `/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5`
- **Planck PR4 Lensing Noise ($N_\ell^{\kappa\kappa}$)**: `/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat`
- **ACT DR6 Lensing Noise ($N_\ell^{\kappa\kappa}$)**: `/global/homes/j/jhawla/thesis-desi-cmb-fli/data/N_L_kk_act_dr6_lensing_v1_baseline.txt`
