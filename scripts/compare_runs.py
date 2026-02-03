#!/usr/bin/env python
"""
Compare multiple runs of run_inference.py using GetDist.
Generates a triangle plot comparing the posterior distributions of parameters.
Supports custom burn-ins, chain exclusions, and displays constraints in the legend.

Usage:
    python scripts/compare_runs.py run1 run2 --burn_ins 0.0 0.0 --excludes "0,1" "" --labels "Run A" "Run B"
"""

import argparse
import os
import sys
from pathlib import Path

import jax
import numpy as np
from getdist import plots

try:
    from scripts.analyze_run import load_and_process_run
except ImportError:
    from analyze_run import load_and_process_run

# Memory optimization for JAX
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="Compare runs with GetDist")
    parser.add_argument("runs", nargs='+', help="Paths to run directories")
    parser.add_argument("--labels", nargs='+', help="Labels for the runs (default: Run 1, Run 2...)")
    parser.add_argument("--burn_ins", nargs='+', type=float, help="Burn-in fraction per run (default: 0.0)")
    parser.add_argument("--exclude_chains", nargs='+', help="Excluded chain indices per run (comma-separated, e.g. '0,1' '3')")
    parser.add_argument("--output", default="figures/comparison_triangle.png", help="Output filename")

    args = parser.parse_args()

    n_runs = len(args.runs)

    # Handle arguments broadcasting
    labels = args.labels if args.labels else [f"Run {i+1}" for i in range(n_runs)]
    if len(labels) != n_runs:
        print(f"Warning: {len(labels)} labels provided for {n_runs} runs. Using defaults for remainder.")
        labels += [f"Run {i+1}" for i in range(len(labels), n_runs)]

    burn_ins = args.burn_ins if args.burn_ins else [0.0] * n_runs
    if len(burn_ins) == 1 and n_runs > 1:
        burn_ins = burn_ins * n_runs
    if len(burn_ins) != n_runs:
         raise ValueError(f"Number of burn-in values ({len(burn_ins)}) must match number of runs ({n_runs}) or be 1.")

    excludes = args.exclude_chains if args.exclude_chains else [""] * n_runs
    if len(excludes) == 1 and n_runs > 1 and excludes[0] == "":
         excludes = [""] * n_runs
    if len(excludes) != n_runs:
         raise ValueError(f"Number of exclude_chains arguments ({len(excludes)}) must match number of runs ({n_runs}).")

    # Process each run
    samples_list = []
    truth_common = None
    all_params = set()
    run_stats = []

    for i, run_dir in enumerate(args.runs):
        # Parse exclude string "1,2" -> [1, 2]
        ex_str = excludes[i]
        if ex_str and ex_str.strip():
            ex_indices = [int(x) for x in ex_str.split(',') if x.strip()]
        else:
            ex_indices = []

        try:
            # Use shared loader
            data = load_and_process_run(
                run_dir,
                burn_in=burn_ins[i],
                exclude_chains=ex_indices
            )

            # Extract what we need
            samples_gd = data["physical_chains"].to_getdist(label=labels[i])
            truth = data["truth_vals"]
            params = data["available_params"]
            physical_samples = data["physical_samples"]

            # Store parameter stats for custom legend
            stats = {}
            for param_name in physical_samples:
                vals = physical_samples[param_name].flatten()
                mean = np.mean(vals)
                sigma = np.std(vals)
                stats[param_name] = (mean, sigma)

            samples_list.append(samples_gd)
            all_params.update(params)
            run_stats.append(stats)

            if truth_common is None:
                truth_common = truth
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # sort params to have meaningful order
    priority = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
    common_params = [p for p in priority if p in all_params]

    print(f"\nGenerating plot with params: {common_params}")

    markers = {k: v for k, v in truth_common.items() if k in common_params}

    # Initialize plotter
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 12
    g.settings.figure_legend_loc = 'upper right'
    # Use distinct colors
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7'] # Accessible palette

    # Assign colors to samples
    for i, s in enumerate(samples_list):
        col = colors[i % len(colors)]
        s.updateSettings({'color': col})


    g.triangle_plot(samples_list, params=common_params, filled=True,
                    title_limit=0, markers=markers)

    # ADD COLORED TITLES TO DIAGONAL PLOTS
    # Iterate over diagonal subplots
    colors_text = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, param in enumerate(common_params):
        ax = g.subplots[i, i]
        if ax is None:
            continue

        # Position titles inside the plot area to avoid clipping
        y_base = 0.75
        y_step = 0.12

        for j, stats in enumerate(run_stats):
            if param in stats:
                mean, sigma = stats[param]
                col = colors_text[j % len(colors_text)]

                # Latex format
                txt = f"${mean:.3f} \pm {sigma:.3f}$"

                # Position: Run 0 at top, Run 1 below
                y_pos = y_base - j * y_step

                ax.text(0.5, y_pos, txt, transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='top',
                        color=col, fontsize=10)

    # Adjust figure to prevent clipping
    g.fig.subplots_adjust(top=0.95)

    # Ensure dir exists

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g.export(str(out_path))
    print(f"✓ Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
