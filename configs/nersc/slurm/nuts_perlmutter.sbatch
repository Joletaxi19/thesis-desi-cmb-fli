#!/bin/bash -l
#SBATCH --job-name=desi-cmb-fli
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:03:00
#SBATCH --output=%x-%j.out

module load python

# Activate the thesis environment (placeholder; sync with configs/nersc/perlmutter.yml)
source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250331-1.0.0/conda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/envs/desi-cmb-fli

which python
python --version

export DESI_LSS_CLUS_DIR="/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
srun python scripts/check_desi_ingest.py --tracer LRG --region NGC --rand-index 0 --zmin 0.6 --zmax 0.8
