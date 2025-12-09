
import os

import jax
import jax.numpy as jnp
import pytest
from numpyro.handlers import seed, trace

from desi_cmb_fli.model import FieldLevelModel, FourierSpaceGaussian, default_config

# Skip all tests in this file if not on NERSC
pytestmark = pytest.mark.skipif(
    not os.path.exists("/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat"),
    reason="Planck data only available on NERSC",
)

def test_fourier_gaussian():
    print("Testing FourierSpaceGaussian...")

    N = 64
    loc = jnp.zeros((N, N))
    var_k = jnp.ones((N, N)) # White noise in Fourier
    # Set DC mode variance to huge to test the fix
    var_k = var_k.at[0, 0].set(1e30)

    dist_test = FourierSpaceGaussian(loc, var_k)

    # Test Sampling
    key = jax.random.PRNGKey(0)
    sample_val = dist_test.sample(key)
    print(f"Sample shape: {sample_val.shape}")

    # Check if real
    if jnp.iscomplexobj(sample_val):
        print("FAILED: Sample is complex.")
    else:
        print("PASSED: Sample is real.")

    # Check mean (should be close to 0, not huge)
    mean_val = jnp.mean(sample_val)
    print(f"Sample mean: {mean_val}")
    if jnp.abs(mean_val) > 1.0:
        print("FAILED: Sample mean is too large (DC mode issue).")
    else:
        print("PASSED: Sample mean is small (DC mode fix works).")

    # Test Log Prob
    lp = dist_test.log_prob(sample_val)
    print(f"Log prob: {lp}")

    # Test in Model Context
    print("\nTesting inside FieldLevelModel...")

    # Mock config
    config = default_config.copy()
    config.update({
        "mesh_shape": (N, N, N),
        "box_shape": (1000.0, 1000.0, 1000.0),
        "cmb_enabled": True,
        "cmb_noise_nell": "/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_nlkk_p.dat", # Absolute path
        "cmb_field_npix": N,
        "cmb_field_size_deg": 10.0,
    })

    model = FieldLevelModel(**config)

    # 1. Generative Mode (Predict)
    print("  Running predict (generative)...")
    try:
        truth = model.predict(samples={"Omega_m": 0.3, "sigma8": 0.8, "b1": 1.0, "b2": 0.0, "bs2": 0.0, "bn2": 0.0}, rng=key)
        if "kappa_obs" in truth:
            print(f"    kappa_obs generated: {truth['kappa_obs'].shape}")
            k_mean = jnp.mean(truth['kappa_obs'])
            print(f"    kappa_obs mean: {k_mean}")
            if jnp.abs(k_mean) > 1.0:
                 print("    FAILED: kappa_obs mean is too large.")
            else:
                 print("    PASSED: Generative mode works and mean is reasonable.")
        else:
            print("    FAILED: kappa_obs not found in truth.")
    except Exception as e:
        print(f"    FAILED: Predict raised exception: {e}")
        import traceback
        traceback.print_exc()

    # 2. Inference Mode (Conditioned)
    print("  Running likelihood (inference)...")
    data = {"obs": truth["obs"], "kappa_obs": truth["kappa_obs"]}

    def conditioned_model():
        model.condition(data)
        return model._model() # Call internal model

    try:
        # Trace the model to check log_prob
        tr = trace(seed(conditioned_model, key)).get_trace()
        if "kappa_obs" in tr:
            log_prob = tr["kappa_obs"]["fn"].log_prob(tr["kappa_obs"]["value"])
            print(f"    kappa_obs log_prob: {log_prob}")
            print("    PASSED: Inference mode works.")
        else:
            print("    FAILED: kappa_obs not found in trace.")
    except Exception as e:
        print(f"    FAILED: Inference raised exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fourier_gaussian()
