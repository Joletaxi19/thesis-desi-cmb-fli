#!/bin/bash -l

# Load environment
module load cudatoolkit/12.4
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli
export LD_LIBRARY_PATH=$(echo ${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}

# Run script with config path (argument 1, default to standard config)
CONFIG_PATH=${1:-configs/05_cmb_lensing/config.yaml}
echo "Using config: $CONFIG_PATH"

python -u scripts/05_cmb_lensing.py --config "$CONFIG_PATH"

# Cleanup temporary config file if it's in the submitted directory
if [[ "$CONFIG_PATH" == *"configs/05_cmb_lensing/submitted/"* ]]; then
    echo "Cleaning up temporary config: $CONFIG_PATH"
    rm "$CONFIG_PATH"
fi
