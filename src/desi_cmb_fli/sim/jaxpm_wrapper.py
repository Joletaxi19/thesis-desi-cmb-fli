"""Minimal, toy simulation utilities based on JAX.

These functions are placeholders for a cosmological simulation pipeline. They
intentionally keep the math simple so they are fast, easy to understand, and
useful for testing the rest of the package (I/O, plotting, wiring, etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_linear_density(
    seed: int,
    mesh_shape: tuple[int, int, int] = (128, 128, 128),
    box_size: float = 200.0,
    sigma8: float = 0.8,
) -> jax.Array:
    """Generate a Gaussian random field representing the linear density contrast.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random number generator.
    mesh_shape : tuple of int, optional
        Number of grid cells along each axis, default ``(128, 128, 128)``. The
        size is small so the example runs quickly.
    box_size : float, optional
        Size of the simulated box in arbitrary units, default ``200.0``. This
        value is arbitrary and chosen for demonstration purposes.
    sigma8 : float, optional
        Placeholder normalization value, default ``0.8``. The amplitude is not
        enforced in this toy model.

    Returns
    -------
    jax.Array
        Random Gaussian field on the specified mesh.

    Notes
    -----
    This routine is a toy example and does not perform any cosmological
    normalization.
    """
    # Use a deterministic PRNG key derived from the provided seed.
    key = jax.random.PRNGKey(seed)
    # Draw a standard normal field on the requested mesh. Single precision is
    # sufficient for the toy example and yields smaller arrays.
    lin = jax.random.normal(key, shape=mesh_shape, dtype=jnp.float32)
    return lin


def nbody_evolve(
    lin: jax.Array,
    a_start: float = 0.02,
    a_final: float = 1.0,
    n_steps: int = 10,
) -> jax.Array:
    """Propagate a density field through a trivial "evolution".

    Parameters
    ----------
    lin : jax.Array
        Linear density field to evolve.
    a_start : float, optional
        Starting scale factor, default ``0.02``, representing an early epoch in
        this toy setup.
    a_final : float, optional
        Final scale factor, default ``1.0``, corresponding to today.
    n_steps : int, optional
        Number of integration steps, default ``10``. This coarse sampling keeps
        runtime short for examples.

    Returns
    -------
    jax.Array
        The input field unchanged.

    Notes
    -----
    This is a placeholder for a real N-body solver; it simply returns the
    input field.
    """
    # Placeholder for a real evolution step; the identity keeps tests simple
    # and makes the data flow explicit in the rest of the pipeline.
    return lin


def mock_summary(state: jax.Array) -> dict[str, float]:
    """Compute a very simple power statistic of the field.

    Parameters
    ----------
    state : jax.Array
        Density field to summarize.

    Returns
    -------
    dict of str to float
        Dictionary containing the mean power of the field.

    Notes
    -----
    The statistic uses a basic FFT and is intended only for testing.
    """
    # Compute a simple, global power proxy via an FFT and average power.
    fft = jnp.fft.fftn(state)
    power = jnp.mean(jnp.abs(fft) ** 2)
    return {"toy_power": float(power)}
