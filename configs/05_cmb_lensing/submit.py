#!/usr/bin/env python3
"""Submit SLURM job for joint galaxies+CMB lensing inference."""

import subprocess
import sys
from pathlib import Path

import yaml

# Load config
config_path = Path("configs/05_cmb_lensing/config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

slurm = cfg["slurm"]

# Build sbatch command
cmd = [
    "sbatch",
    f"--job-name={slurm['job_name']}",
    f"--account={slurm['account']}",
    f"--qos={slurm['qos']}",
    f"--constraint={slurm['constraint']}",
    f"--nodes={slurm['nodes']}",
    f"--time={slurm['time']}",
    "--output=logs/%x-%j.out",
]

# Add GPUs only if constraint is gpu
if slurm["constraint"] == "gpu":
    cmd.append(f"--gpus={slurm['gpus']}")

# The actual script to run
cmd.append("configs/05_cmb_lensing/run.sbatch")

# Submit
print(f"Submitting job with: {' '.join(cmd)}")
result = subprocess.run(cmd)
sys.exit(result.returncode)
