#!/bin/bash -l
#SBATCH --job-name=desi-cmb-fli
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Update this path if you relocate the YAML file.
CONFIG_FILE="$(dirname "$0")/../perlmutter.yml"

module load python
module load cudatoolkit

# Activate the thesis environment (placeholder; sync with configs/nersc/perlmutter.yml)
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli

# Provide the configuration to the Python entry point as an environment variable.
export DESI_CMB_FLI_CONFIG="${CONFIG_FILE}"

echo "Using configuration: ${DESI_CMB_FLI_CONFIG}"
which python
python --version

# Replace the module invocation below once the inference driver exists.
python -m desi_cmb_fli --config "${DESI_CMB_FLI_CONFIG}"
