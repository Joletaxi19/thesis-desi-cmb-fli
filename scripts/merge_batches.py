
import argparse
from pathlib import Path

import numpy as np


def merge_batches(run_dir):
    """
    Merge separate batch output files into a single samples.npz.

    Args:
        run_dir (Path or str): Path to the run directory.

    Returns:
        dict: The merged samples dictionary if successful, None if no batches found.
    """
    run_dir = Path(run_dir)
    config_dir = run_dir / "config"
    batches = sorted(config_dir.glob("samples_batch_*.npz"))

    if not batches:
        print("No batches found.")
        return None

    # Sort numerically by batch index
    batches.sort(key=lambda p: int(p.stem.split('_')[-1]))

    print(f"Found {len(batches)} batches.")

    samples = {}

    for batch_file in batches:
        print(f"Loading {batch_file.name}...")
        data = np.load(batch_file)
        for k in data.files:
            if k not in samples:
                samples[k] = []
            samples[k].append(data[k])

    # Concatenate
    merged = {}
    for k, v in samples.items():
        # v is list of (n_chains, n_samples)
        # We need to concat along axis 1 (n_samples)
        merged[k] = np.concatenate(v, axis=1)
        print(f"Merged {k}: {merged[k].shape}")

    out_path = config_dir / "samples.npz"
    np.savez(out_path, **merged)
    print(f"Saved {out_path}")

    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    args = parser.parse_args()

    merge_batches(args.run_dir)

if __name__ == "__main__":
    main()
