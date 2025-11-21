#!/bin/bash -l

# Load environment
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}

# Run script with config path
python -u scripts/04_field_level_inference.py --config configs/04_field_level_inference/config.yaml
