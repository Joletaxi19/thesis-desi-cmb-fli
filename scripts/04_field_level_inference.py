#!/usr/bin/env python

# # Field-Level Inference on Synthetic Data
#
# This notebook demonstrates the complete field-level inference pipeline:
# 1. Create a `FieldLevelModel` from configuration
# 2. Generate synthetic observations from truth parameters
# 3. Run MCMC inference to recover parameters
# 4. Analyze chains and compare to truth
# 5. Visualize results
#
# This follows the approach from [benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level),
# using the MCLMC sampler (already implemented in benchmark) instead of NUTS.

# In[2]:


import argparse
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import yaml
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from jax import jit, vmap

from desi_cmb_fli.model import FieldLevelModel, default_config
from desi_cmb_fli.samplers import get_mclmc_run, get_mclmc_warmup

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Field-Level Inference")
parser.add_argument(
    "--config",
    default="configs/04_field_level_inference/config.yaml",
    help="Path to configuration YAML file",
)
args = parser.parse_args()

# Load configuration
config_path = Path(args.config)
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Create output directory for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(f"outputs/run_{timestamp}")
fig_dir = run_dir / "figures"
config_dir = run_dir / "config"

for d in [fig_dir, config_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Save a copy of the configuration
shutil.copy(config_path, config_dir / "config.yaml")

print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Run directory: {run_dir}")

# Check available devices
devices = jax.devices()
print(f"\nAvailable JAX devices: {len(devices)}")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device}")


# ## 1. Model Configuration
#
# We'll set up the field-level model with desired parameters.

# In[ ]:


# Create configuration from YAML
config = default_config.copy()
config["mesh_shape"] = tuple(cfg["model"]["mesh_shape"])
config["box_shape"] = tuple(cfg["model"]["box_shape"])
config["evolution"] = cfg["model"]["evolution"]
config["lpt_order"] = cfg["model"]["lpt_order"]
config["a_obs"] = cfg["model"]["a_obs"]
config["gxy_density"] = cfg["model"]["gxy_density"]

# Remove invalid keys that may be in default_config
config.pop("scale_factor", None)
config.pop("ng", None)

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")


# In[6]:


# Create model
model = FieldLevelModel(**config)
print(model)


# ## 2. Generate Synthetic Data
#
# We'll create a "truth" dataset by forward-simulating the model with known parameters.

# In[7]:


# Truth parameters from config
truth_params = cfg["truth_params"]

print("Truth parameters:")
for k, v in truth_params.items():
    print(f"  {k}: {v}")


# In[8]:


# Generate truth data
seed = cfg["seed"]
rng = jr.key(seed)
print(f"Random seed: {seed}")

truth = model.predict(samples=truth_params, hide_base=False, frombase=True, rng=rng)

obs_data = truth["obs"]
print(f"Observed field shape: {obs_data.shape}")
print(f"Mean galaxy count: {jnp.mean(obs_data):.4f}")
print(f"Std: {jnp.std(obs_data):.4f}")


# In[9]:


# Visualize observed field (slice through center)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Use mesh_shape//2 for center slices
center_idx = config["mesh_shape"][0] // 2
slices = [
    obs_data[center_idx, :, :],
    obs_data[:, center_idx, :],
    obs_data[:, :, center_idx],
]
titles = [f"YZ slice (x={center_idx})", f"XZ slice (y={center_idx})", f"XY slice (z={center_idx})"]

for ax, data, title in zip(axes, slices, titles, strict=True):
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    plt.colorbar(im, ax=ax, label="Galaxy count")

plt.tight_layout()
plt.savefig(fig_dir / "obs_field_slices.png")
plt.close()


# ## 3. Run MCMC Inference
#
# Now we'll use MCLMC to infer the parameters from the observed data.

# In[10]:


# Reset and condition model on observations
model.reset()
model.condition({"obs": obs_data})


# In[11]:


# MCMC configuration from config
num_warmup = cfg["mcmc"]["num_warmup"]
num_samples = cfg["mcmc"]["num_samples"]
num_chains = cfg["mcmc"]["num_chains"]

# Initialize from Kaiser posterior - MULTIPLE positions for chain diversity
rng_init = jr.key(seed + 1)
init_keys = jr.split(rng_init, num_chains)

# Vectorized initialization (one init per chain)
print(f"\nGenerating {num_chains} diverse initial positions...")
init_params = jit(vmap(partial(model.kaiser_post, obs_data=obs_data, base=False)))(init_keys)

print("Initialized parameters (showing first chain):")
for k, v in init_params.items():
    if not k.startswith("init_mesh"):
        print(f"  {k}: shape {v.shape}, first={v[0]:.6f}")


# In[ ]:

# MCLMC specific parameters
mclmc_config = cfg["mcmc"].get("mclmc", {})
desired_energy_var = float(
    mclmc_config.get("desired_energy_var", 5e-4)
)  # Convert to float in case YAML parsed as string
diagonal_preconditioning = bool(mclmc_config.get("diagonal_preconditioning", False))
thinning = int(mclmc_config.get("thinning", 1))

print("\nMCMC settings:")
print("  Sampler: MCLMC (with jit+vmap parallelization)")
print(f"  Warmup: {num_warmup}")
print(f"  Samples: {num_samples}")
print(f"  Chains: {num_chains}")
print(f"  Desired energy var: {desired_energy_var}")
print(f"  Diagonal preconditioning: {diagonal_preconditioning}")
print(f"  Thinning: {thinning}")

# Check parallelization
if len(devices) >= num_chains:
    print(f"\n✓ {num_chains} chains will run in PARALLEL across {len(devices)} GPU(s)")
else:
    print(f"\n⚠️  {len(devices)} GPU(s) for {num_chains} chains - vmap will distribute optimally")


# In[13]:


# Run MCLMC with PARALLEL execution using jit+vmap
print("\n🚀 Starting PARALLEL MCLMC sampling...")

# Use the model's built-in logpdf method
logdf = model.logpdf

# Warmup phase - PARALLELIZED
if num_warmup > 0:
    print(f"  Warmup ({num_warmup} steps, {num_chains} chains in parallel)...")
    warmup_fn = jit(
        vmap(
            get_mclmc_warmup(
                logdf=logdf,
                n_steps=num_warmup,
                config=None,
                desired_energy_var=desired_energy_var,
                diagonal_preconditioning=diagonal_preconditioning,
            )
        )
    )

    rng_warmup = jr.key(seed + 2)
    warmup_keys = jr.split(rng_warmup, num_chains)
    state, config = warmup_fn(warmup_keys, init_params)

    print("  ✓ Warmup completed")
    print(f"    Median logdensity: {jnp.median(state.logdensity):.2f}")
    median_L_adapted = jnp.median(config.L)
    print(f"    Adapted L (median): {median_L_adapted:.6f}")
    print(f"    Median step_size: {jnp.median(config.step_size):.6f}")

    # Recalculate L based on efficiency criterion (as in benchmark-field-level)
    # See: https://github.com/hsimonfroy/benchmark-field-level/blob/main/examples/infer_model.ipynb#L870
    eval_per_ess = 1e3  # Target: 1000 gradient evals per effective sample
    median_step_size = jnp.median(config.step_size)
    median_imm = jnp.median(config.inverse_mass_matrix, axis=0)

    # Formula: L = factor * (eval_per_ess / 2) * step_size
    # Factor 0.4 for MCLMC (empirical, from benchmark)
    recalculated_L = 0.4 * eval_per_ess / 2 * median_step_size

    config = MCLMCAdaptationState(
        L=jnp.full(num_chains, recalculated_L),
        step_size=jnp.full(num_chains, median_step_size),
        inverse_mass_matrix=jnp.tile(median_imm, (num_chains, 1)),
    )

    print(f"    Recalculated L for efficiency: {recalculated_L:.6f} (was: {median_L_adapted:.6f})")
else:
    # Initialize without warmup
    from blackjax.mcmc.mclmc import init as mclmc_init

    rng_init_sampling = jr.key(seed + 2)
    init_keys = jr.split(rng_init_sampling, num_chains)
    state = jit(vmap(lambda k, p: mclmc_init(position=p, logdensity_fn=logdf, rng_key=k)))(
        init_keys, init_params
    )
    config = None

# Sampling phase - PARALLELIZED
print(f"  Sampling ({num_samples} steps, {num_chains} chains in parallel)...")
run_fn = jit(
    vmap(
        get_mclmc_run(
            logdf=logdf,
            n_samples=num_samples,
            transform=None,
            thinning=thinning,
            progress_bar=True,
        )
    )
)

rng_sample = jr.key(seed + 3)
sample_keys = jr.split(rng_sample, num_chains)
state, samples_dict = run_fn(sample_keys, state, config)

print("✓ MCMC sampling completed!")

# Organize samples by parameter (already properly shaped from vmap)
samples = {}
param_names = [k for k in samples_dict.keys() if k not in ["logdensity", "mse_per_dim", "n_evals"]]
for param in param_names:
    samples[param] = samples_dict[param]

# Store info fields separately
all_infos = [
    {
        "logdensity": samples_dict["logdensity"][i] if "logdensity" in samples_dict else None,
        "mse_per_dim": samples_dict["mse_per_dim"][i] if "mse_per_dim" in samples_dict else None,
        "n_evals": samples_dict["n_evals"][i] if "n_evals" in samples_dict else None,
    }
    for i in range(num_chains)
]


# ## 4. Analyze Results
#
# Check convergence diagnostics and compare recovered parameters to truth.

# In[14]:

# Print MCMC diagnostics
print("\nSampling statistics:")
for param in param_names:
    values = samples[param].reshape(-1)
    print(f"  {param}:")
    print(f"    Mean: {np.mean(values):.6f}")
    print(f"    Std:  {np.std(values):.6f}")

print("\nDiagnostics:")
if "mse_per_dim" in samples_dict:
    for i_chain in range(num_chains):
        mse = np.mean(samples_dict["mse_per_dim"][i_chain])
        print(f"  Chain {i_chain}: MSE per dim = {mse:.6e}")


# In[15]:


# Compare to truth
comparison_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]

print("\nParameter recovery:")
print(f"{'Parameter':<10} {'Truth':<10} {'Mean':<10} {'Std':<10} {'Bias (σ)':<10}")
print("-" * 60)

for param in comparison_params:
    param_key = param + "_"
    if param_key in samples:
        chain_samples = samples[param_key].reshape(-1)  # Flatten chains
        mean = np.mean(chain_samples)
        std = np.std(chain_samples)
        truth_val = truth_params[param]
        bias_sigma = (mean - truth_val) / std if std > 0 else 0

        print(f"{param:<10} {truth_val:<10.4f} {mean:<10.4f} {std:<10.4f} {bias_sigma:<10.2f}")


# ## 5. Visualize Chains
#
# Plot trace plots and posterior distributions.

# In[16]:


# Trace plots
fig, axes = plt.subplots(len(comparison_params), 1, figsize=(12, 2 * len(comparison_params)))

for i, param in enumerate(comparison_params):
    param_key = param + "_"
    if param_key in samples:
        ax = axes[i]
        chain_data = samples[param_key]

        # Plot each chain
        for chain_idx in range(num_chains):
            ax.plot(
                chain_data[chain_idx, :], alpha=0.7, label=f"Chain {chain_idx}" if i == 0 else ""
            )

        # Truth line
        ax.axhline(
            truth_params[param],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Truth" if i == 0 else "",
        )

        ax.set_ylabel(param)
        ax.set_xlabel("Iteration")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "trace_plots.png")
plt.close()


# In[17]:


# Posterior distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, param in enumerate(comparison_params):
    param_key = param + "_"
    if param_key in samples:
        ax = axes[i]
        chain_samples = samples[param_key].reshape(-1)

        # Histogram
        ax.hist(chain_samples, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")

        # Truth line
        ax.axvline(truth_params[param], color="red", linestyle="--", linewidth=2, label="Truth")

        # Mean line
        mean_val = np.mean(chain_samples)
        ax.axvline(mean_val, color="green", linestyle="-", linewidth=2, label="Mean")

        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "posterior_distributions.png")
plt.close()


# ## 6. Corner Plot
#
# Show joint posterior distributions.

# In[18]:


try:
    import corner

    # Prepare data for corner plot - use comparison_params with underscore suffix
    corner_labels = []
    corner_data_list = []
    corner_truths = []

    for param in comparison_params:
        param_key = param + "_"
        if param_key in samples:
            corner_labels.append(param)
            corner_data_list.append(samples[param_key].reshape(-1))
            corner_truths.append(truth_params[param])

    corner_data = np.column_stack(corner_data_list)

    fig = corner.corner(
        corner_data,
        labels=corner_labels,
        truths=corner_truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.savefig(fig_dir / "corner_plot.png")
    plt.close()

except ImportError:
    print("Corner plot requires the 'corner' package. Install with: pip install corner")


# ## Summary
#
# This script demonstrated:
# - Creating a field-level model with specified configuration (Gaussian likelihood as in benchmark)
# - Generating synthetic observations from truth parameters
# - Running MCMC inference with MCLMC sampler (parallel GPU execution)
# - Analyzing convergence and parameter recovery
# - Visualizing chains and posteriors
#
# Note: Uses MCLMC sampler (already implemented in benchmark) instead of NUTS.
