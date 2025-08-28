from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable
from ..sim.jaxpm_wrapper import make_linear_density, nbody_evolve, mock_summary

def toy_fli(seed: int = 0) -> Dict[str, Any]:
    """Minimal toy FLI loop: sample initial field, evolve, compute summary."""
    lin = make_linear_density(seed)
    state = nbody_evolve(lin)
    summ = mock_summary(state)
    return {"summary": summ}
