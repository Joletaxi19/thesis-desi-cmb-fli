from __future__ import annotations

from typing import Any

from ..sim.jaxpm_wrapper import make_linear_density, mock_summary, nbody_evolve


def toy_fli(seed: int = 0) -> dict[str, Any]:
    """Minimal toy FLI loop: sample initial field, evolve, compute summary."""
    lin = make_linear_density(seed)
    state = nbody_evolve(lin)
    summ = mock_summary(state)
    return {"summary": summ}
