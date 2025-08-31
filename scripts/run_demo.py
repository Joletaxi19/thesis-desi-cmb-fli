"""Minimal demo for the toy field-level inference pipeline.

Runs the tiny FLI loop, logs a scalar summary, and saves a small plot.
Kept deliberately minimal so that new users can read it top-to-bottom and
understand how the package fits together.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from desi_cmb_fli.analysis.plots import save_toy_plot
from desi_cmb_fli.inference.fli import toy_fli
from desi_cmb_fli.utils.logging import setup_logger

log = setup_logger()


def parse_args() -> Namespace:
    """Define and parse CLI options for the demo run."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the toy FLI run (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/toy_summary.png"),
        help="Path to save the summary plot (default: figures/toy_summary.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Run the miniature FLI pipeline and extract a scalar summary.
    results = toy_fli(seed=args.seed)
    val = results["summary"]["toy_power"]
    log.info(f"Toy power: {val:.3e}")

    # Produce a tiny plot for quick visual feedback and write it to disk.
    save_toy_plot(val, args.output)
    log.info(f"Saved {args.output}")


if __name__ == "__main__":
    main()
