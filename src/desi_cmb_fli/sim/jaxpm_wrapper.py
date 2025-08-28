from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def make_linear_density(seed: int, mesh_shape=(128, 128, 128), box_size=200.0, sigma8=0.8):
    """
    Toy: génère un champ gaussien normal (pas de normalisation sigma8 réelle ici).
    On évite volontairement fftk/pfft3d pour que le demo reste ultra-simple.
    """
    key = jax.random.PRNGKey(seed)
    lin = jax.random.normal(key, shape=mesh_shape, dtype=jnp.float32)
    return lin


def nbody_evolve(lin, a_start=0.02, a_final=1.0, n_steps=10):
    """
    Toy: identité (pas de dynamique réelle).
    Remplace plus tard par un vrai wrapper JAX-PM (LPT/N-body).
    """
    return lin


def mock_summary(state) -> dict[str, Any]:
    """
    Toy: calcule une "puissance" très simplifiée via FFT numpy de JAX.
    """
    fft = jnp.fft.fftn(state)
    power = jnp.mean(jnp.abs(fft) ** 2)
    return {"toy_power": float(power)}
