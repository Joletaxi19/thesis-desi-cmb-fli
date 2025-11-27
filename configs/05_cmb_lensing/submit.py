#!/usr/bin/env python3
"""Submit SLURM job for joint galaxies+CMB lensing inference."""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Load config
config_path = Path("configs/05_cmb_lensing/config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

slurm = cfg["slurm"]

# Create a unique copy of the config for this job
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submitted_dir = config_path.parent / "submitted"
submitted_dir.mkdir(exist_ok=True)
job_config_path = submitted_dir / f"config_{timestamp}.yaml"

print(f"Copying config to: {job_config_path}")
shutil.copy(config_path, job_config_path)

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

# The actual script to run, passing the specific config file
cmd.append("configs/05_cmb_lensing/run.sbatch")
cmd.append(str(job_config_path))

# Submit
print(f"Submitting job with: {' '.join(cmd)}")
result = subprocess.run(cmd)
sys.exit(result.returncode)
