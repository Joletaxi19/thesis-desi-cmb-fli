from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint
from jaxpm.kernels import fftk
from jaxpm.pm import linear_field, lpt, make_ode_fn, pm_forces
from typing import Dict, Any

def make_linear_density(seed: int, mesh_shape=(128,128,128), box_size=200.0, sigma8=0.8):
    key = jax.random.PRNGKey(seed)
    kvec = fftk(mesh_shape)
    lin = linear_field(mesh_shape, box_size, key)
    # Note: Placeholder; in practice, normalize to sigma8, set cosmology with jax-cosmo etc.
    return lin

def nbody_evolve(lin, a_start=0.02, a_final=1.0, n_steps=10):
    # Very light stub; replace with a correct cosmology + growth
    state = lin  # placeholder
    return state

def mock_summary(state) -> Dict[str, Any]:
    # Return toy "summary statistics"
    power = jnp.mean(jnp.abs(jnp.fft.fftn(state))**2)
    return {"toy_power": float(power)}
