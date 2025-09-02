"""End-to-end demo (YAML-config only): synthetic maps → FLI vs 3×2pt.

This script generates small flat-sky maps on CPU, runs a field-level Fourier
likelihood and a 3×2pt binned power-spectrum likelihood, and plots their
posterior contours together with the true parameters used for generation.

Configuration is provided via a YAML file only, for reproducibility and
cleanliness. See configs/demo.yaml for an example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from desi_cmb_fli.inference.fli_flat import fli_fourier_likelihood
from desi_cmb_fli.inference.three_by_two import three_by_two_likelihood
from desi_cmb_fli.sim.flat_sky import SynthConfig, generate_maps
from desi_cmb_fli.utils.logging import setup_logger

log = setup_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="YAML config path (required)")
    return p.parse_args()


def _grid_from_seq(seq) -> np.ndarray:
    amin, amax, n = float(seq[0]), float(seq[1]), int(seq[2])
    return np.linspace(amin, amax, n)


def main() -> None:
    args = parse_args()
    # Load YAML config (required)
    with open(args.config) as f:
        yml = yaml.safe_load(f) or {}

    def yget(path, default):
        cur = yml
        for key in path:
            if cur is None or key not in cur:
                return default
            cur = cur[key]
        return cur if cur is not None else default

    # Helper function to safely extract values, handling both scalar and dict cases
    def safe_get(path, default, value_type=None):
        val = yget(path, default)
        if isinstance(val, dict):
            # If it's a dict, try to get the value using the last key in the path
            key = path[-1] if path else "value"
            val = val.get(key, default)
        return value_type(val) if value_type else val

    # Resolve simulation params from YAML with cleaner extraction
    nx = safe_get(["sim", "nx"], 128, int)
    ny = safe_get(["sim", "ny"], 128, int)
    box_size = safe_get(["sim", "box_size"], 400.0, float)
    A_true = safe_get(["sim", "A"], 1.0, float)
    b_true = safe_get(["sim", "b"], 1.2, float)
    sigma_g = safe_get(["sim", "sigma_g"], 0.5, float)
    sigma_gamma = safe_get(["sim", "sigma_gamma"], 0.3, float)
    seed = safe_get(["sim", "seed"], 0, int)

    # Parameter grids from YAML
    A_vals = _grid_from_seq(yget(["inference", "A_grid"], [0.5, 2.0, 60]))
    b_vals = _grid_from_seq(yget(["inference", "b_grid"], [0.5, 2.0, 60]))
    nbin = safe_get(["inference", "nbin"], 15, int)

    out_path_val = safe_get(["output", "path"], "figures/synthetic_constraints.png", str)
    out_path = Path(out_path_val)

    cfg = SynthConfig(
        nx=nx,
        ny=ny,
        box_size=box_size,
        A=A_true,
        b=b_true,
        sigma_g=sigma_g,
        sigma_gamma=sigma_gamma,
        seed=seed,
    )
    log.info(
        "Generating synthetic maps with "
        f"{{'nx': {nx}, 'ny': {ny}, 'box_size': {box_size}, 'A': {A_true}, 'b': {b_true}, "
        f"'sigma_g': {sigma_g}, 'sigma_gamma': {sigma_gamma}, 'seed': {seed}}}"
    )
    data = generate_maps(cfg)

    g = data["g"]
    E = data["gammaE"]
    kk = data["kk"]

    log.info("Running field-level Fourier-mode likelihood (grid scan)...")
    fli_res = fli_fourier_likelihood(
        g,
        E,
        box_size=cfg.box_size,
        sigma_g=cfg.sigma_g,
        sigma_gamma=cfg.sigma_gamma,
        A_grid=A_vals,
        b_grid=b_vals,
    )

    log.info("Running 3×2pt binned spectra likelihood (grid scan)...")
    tbt_res = three_by_two_likelihood(
        g,
        E,
        kk,
        box_size=cfg.box_size,
        sigma_g=cfg.sigma_g,
        sigma_gamma=cfg.sigma_gamma,
        A_grid=A_vals,
        b_grid=b_vals,
        nbin=nbin,
    )

    # Convert NLL to delta-chi2 for contours
    def delta_nll(nll: np.ndarray) -> np.ndarray:
        return nll - nll.min()

    d_fli = delta_nll(fli_res.nll)
    d_tbt = delta_nll(tbt_res.nll)

    # Posterior grids (unnormalized), then 1D marginals
    Pf = np.exp(-d_fli)
    Pt = np.exp(-d_tbt)
    pA_f = Pf.sum(axis=1)
    pA_t = Pt.sum(axis=1)
    pb_f = Pf.sum(axis=0)
    pb_t = Pt.sum(axis=0)
    # Normalize for plotting
    pA_f /= pA_f.max() if pA_f.max() > 0 else 1
    pA_t /= pA_t.max() if pA_t.max() > 0 else 1
    pb_f /= pb_f.max() if pb_f.max() > 0 else 1
    pb_t /= pb_t.max() if pb_t.max() > 0 else 1

    # Plot: 2x2 grid (contours + 1D marginals)
    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1, 1], wspace=0.3)
    ax2d = fig.add_subplot(gs[:, 0])
    axA = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])

    Agrid, Bgrid = np.meshgrid(A_vals, b_vals, indexing="xy")
    # Contours
    ax2d.contour(
        Agrid, Bgrid, d_fli, levels=[2.30, 6.17], colors=["C0", "C0"], linestyles=["-", "--"]
    )
    ax2d.contour(
        Agrid, Bgrid, d_tbt, levels=[2.30, 6.17], colors=["C1", "C1"], linestyles=["-", "--"]
    )
    ax2d.scatter([cfg.A], [cfg.b], c="k", s=30, label="Truth")
    ax2d.set_xlabel("Amplitude A")
    ax2d.set_ylabel("Bias b")
    ax2d.set_title("FLI (C0) vs 3×2pt (C1)")
    ax2d.legend(loc="best")

    # 1D marginals
    axA.plot(A_vals, pA_f, color="C0", label="FLI")
    axA.plot(A_vals, pA_t, color="C1", label="3×2pt")
    axA.axvline(cfg.A, color="k", lw=1, ls=":")
    axA.set_ylabel("p(A) [arb]")
    axA.legend(loc="best")

    axb.plot(b_vals, pb_f, color="C0", label="FLI")
    axb.plot(b_vals, pb_t, color="C1", label="3×2pt")
    axb.axvline(cfg.b, color="k", lw=1, ls=":")
    axb.set_xlabel("b")
    axb.set_ylabel("p(b) [arb]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"Saved constraints figure to {out_path}")

    log.info(f"FLI ML estimate: A={fli_res.A_ML:.3f}, b={fli_res.b_ML:.3f}")
    log.info(f"3×2pt ML estimate: A={tbt_res.A_ML:.3f}, b={tbt_res.b_ML:.3f}")


if __name__ == "__main__":
    main()
