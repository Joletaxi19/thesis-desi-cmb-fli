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
from matplotlib.lines import Line2D

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

    # Define consistent color palette
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']

    # Process each run
    all_data = []
    for i, run_dir in enumerate(args.runs):
        # Parse exclude string
        ex_str = excludes[i]
        if ex_str and ex_str.strip():
            ex_indices = [int(x) for x in ex_str.split(',') if x.strip()]
        else:
            ex_indices = []

        try:
            data = load_and_process_run(run_dir, burn_in=burn_ins[i], exclude_chains=ex_indices)

            # Get available params (already filtered for CMB-only)
            params = data["available_params"]
            physical_samples = data["physical_samples"]

            # Create GetDist sample with label (color assigned later)
            samples_gd = data["physical_chains"].to_getdist(label=labels[i])

            all_data.append({
                'samples': samples_gd,
                'params': set(params),
                'truth': data["truth_vals"],
                'physical_samples': physical_samples,
                'color': colors[i % len(colors)],
                'label': labels[i]
            })

        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Determine parameters
    priority = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
    all_params = set()
    for d in all_data:
        all_params.update(d['params'])

    ordered_params = [p for p in priority if p in all_params]
    extra_params = sorted([p for p in all_params if p not in ordered_params])
    ordered_params += extra_params

    # Find base run (most parameters)
    max_idx = max(range(n_runs), key=lambda i: len(all_data[i]['params']))
    base_params = [p for p in ordered_params if p in all_data[max_idx]['params']]

    print(f"\nBase params for triangle plot: {base_params}")

    # Separate full runs (all params) vs partial (CMB-only with Omega_m/sigma8)
    full_runs = [i for i in range(n_runs) if all(p in all_data[i]['params'] for p in base_params)]
    partial_runs = [i for i in range(n_runs) if i not in full_runs]

    print(f"Full runs: {[labels[i] for i in full_runs]}")
    print(f"Partial runs: {[labels[i] for i in partial_runs]}")

    # Truth markers
    markers = {k: v for k, v in all_data[max_idx]['truth'].items() if k in base_params}

    # Initialize plotter
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 12
    g.settings.num_plot_contours = 2

    # Prepare full runs with explicit colors and transparency
    samples_full = []
    colors_full = []
    contour_args_list = []
    for idx in full_runs:
        samples_full.append(all_data[idx]['samples'])
        colors_full.append(all_data[idx]['color'])
        # No transparency for full runs
        contour_args_list.append({'alpha': 1.0})

    # Create triangle plot with explicit color specification
    g.triangle_plot(samples_full, params=base_params, filled=True,
                    title_limit=0, markers=markers,
                    contour_colors=colors_full, line_args=[{'color': c, 'lw': 1.5} for c in colors_full],
                    contour_args=contour_args_list)

    # Overlay partial runs on Omega_m and sigma8
    overlay_params = ["Omega_m", "sigma8"]
    if all(p in base_params for p in overlay_params):
        for run_idx in partial_runs:
            if not all(p in all_data[run_idx]['params'] for p in overlay_params):
                continue

            samples = all_data[run_idx]['samples']
            color = all_data[run_idx]['color']

            # 2D contour - CMB only with transparency (orange)
            x_idx = base_params.index(overlay_params[0])
            y_idx = base_params.index(overlay_params[1])
            ax_2d = g.subplots[y_idx, x_idx]
            g.add_2d_contours(samples, overlay_params[0], overlay_params[1],
                            ax=ax_2d, filled=True, color=color, alpha=0.35)

            # 1D curves
            for param in overlay_params:
                param_idx = base_params.index(param)
                ax_1d = g.subplots[param_idx, param_idx]
                g.add_1d(samples, param, ax=ax_1d, color=color, lw=1.5)

    # Add mean ± std text on diagonal
    for i, param in enumerate(base_params):
        ax = g.subplots[i, i]
        if ax is None:
            continue

        y_base = 0.75
        y_step = 0.12
        text_count = 0

        # Full runs first
        for run_idx in full_runs:
            if param not in all_data[run_idx]['physical_samples']:
                continue
            vals = all_data[run_idx]['physical_samples'][param].flatten()
            mean, sigma = np.mean(vals), np.std(vals)
            color = all_data[run_idx]['color']

            txt = f"${mean:.3f} \pm {sigma:.3f}$"
            y_pos = y_base - text_count * y_step
            ax.text(0.5, y_pos, txt, transform=ax.transAxes,
                    horizontalalignment='center', verticalalignment='top',
                    color=color, fontsize=10)
            text_count += 1

        # Then partial runs
        for run_idx in partial_runs:
            if param not in all_data[run_idx]['physical_samples']:
                continue
            vals = all_data[run_idx]['physical_samples'][param].flatten()
            mean, sigma = np.mean(vals), np.std(vals)
            color = all_data[run_idx]['color']

            txt = f"${mean:.3f} \pm {sigma:.3f}$"
            y_pos = y_base - text_count * y_step
            ax.text(0.5, y_pos, txt, transform=ax.transAxes,
                    horizontalalignment='center', verticalalignment='top',
                    color=color, fontsize=10)
            text_count += 1

    for legend in list(g.fig.legends):
        legend.remove()

    legend_handles = [
        Line2D([0], [0], color=all_data[i]['color'], lw=2, label=all_data[i]['label'])
        for i in range(n_runs)
    ]
    g.fig.legend(handles=legend_handles, loc='upper right', fontsize=12)
    g.fig.subplots_adjust(top=0.95)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.export(str(out_path))
    print(f"✓ Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
