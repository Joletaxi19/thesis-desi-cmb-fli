#!/bin/bash -l
#SBATCH --job-name=desi-cmb-fli
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Load CUDA drivers and activate environment
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}

# Your analysis code here
srun python your_script.py
