from desi_cmb_fli.inference.fli import toy_fli
from desi_cmb_fli.analysis.plots import save_toy_plot
from desi_cmb_fli.utils.logging import setup_logger
from pathlib import Path

log = setup_logger()

def main():
    results = toy_fli(seed=0)
    val = results["summary"]["toy_power"]
    log.info(f"Toy power: {val:.3e}")
    save_toy_plot(val, Path("figures/toy_summary.png"))
    log.info("Saved figures/toy_summary.png")

if __name__ == "__main__":
    main()
