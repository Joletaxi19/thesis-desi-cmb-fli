"""Toy field-level inference (FLI) pipeline wiring.

This module demonstrates the minimal flow of a field-level inference loop:
sample an initial field, evolve it forward, and compute a small set of summary
statistics. The goal is to keep the control flow obvious and the return
structure self-descriptive for easy downstream use.
"""

from __future__ import annotations

from typing import Any

from ..sim.jaxpm_wrapper import make_linear_density, mock_summary, nbody_evolve


def toy_fli(seed: int = 0) -> dict[str, Any]:
    """Run a tiny FLI loop and return summary statistics.

    Parameters
    ----------
    seed
        Random seed used to generate the initial linear field.

    Returns
    -------
    dict
        Dictionary with a single key ``"summary"`` mapping to a dictionary of
        scalar summaries (currently ``{"toy_power": float}``).
    """
    # 1) Sample an initial linear density field from a Gaussian prior.
    lin = make_linear_density(seed)
    # 2) Evolve the field forward using a (placeholder) dynamical model.
    state = nbody_evolve(lin)
    # 3) Reduce the state to simple, test-friendly summary statistics.
    summ = mock_summary(state)
    return {"summary": summ}
