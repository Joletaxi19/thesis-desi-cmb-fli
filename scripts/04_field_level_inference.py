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
# This follows the approach from [benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level).

# In[2]:


import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from desi_cmb_fli.model import FieldLevelModel, default_config

print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"Default backend: {jax.default_backend()}")


# ## 1. Model Configuration
#
# We'll set up the field-level model with desired parameters.

# In[ ]:


# Create configuration
config = default_config.copy()
config["mesh_shape"] = (128, 128, 128)
config["box_shape"] = (200.0, 200.0, 200.0)  # Mpc/h
config["evolution"] = "lpt"
config["lpt_order"] = 2
config["a_obs"] = 0.5  # Scale factor for observation
config["gxy_density"] = 0.001  # Galaxy number density [h³/Mpc³]

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


# Truth parameters
truth_params = {
    "Omega_m": 0.3,
    "sigma8": 0.8,
    "b1": 1.0,
    "b2": 0.0,
    "bs2": 0.0,
    "bn2": 0.0,
}

print("Truth parameters:")
for k, v in truth_params.items():
    print(f"  {k}: {v}")


# In[8]:


# Generate truth data
seed = 42
rng = jr.key(seed)

truth = model.predict(samples=truth_params, hide_base=False, frombase=True, rng=rng)

obs_data = truth["obs"]
print(f"Observed field shape: {obs_data.shape}")
print(f"Mean N_obs: {jnp.mean(obs_data):.4f}")
print(f"Std: {jnp.std(obs_data):.4f}")


# In[9]:


# Visualize observed field (slice through center)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

slices = [
    obs_data[16, :, :],
    obs_data[:, 16, :],
    obs_data[:, :, 16],
]
titles = ["YZ slice (x=16)", "XZ slice (y=16)", "XY slice (z=16)"]

for ax, data, title in zip(axes, slices, titles, strict=True):
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    plt.colorbar(im, ax=ax, label="Galaxy count")

plt.tight_layout()
plt.show()


# ## 3. Run MCMC Inference
#
# Now we'll use NUTS to infer the parameters from the observed data.

# In[10]:


# Reset and condition model on observations
model.reset()
model.condition({"obs": obs_data})


# In[11]:


# Initialize from Kaiser posterior
rng_init = jr.key(123)
init_params = model.kaiser_post(rng_init, obs_data, base=True)

print("Initialized parameters:")
for k, v in init_params.items():
    if k != "init_mesh":
        print(f"  {k}: {v}")


# In[ ]:


# MCMC configuration
num_warmup = 100 # 100 for test, 2000 for real runs
num_samples = 100 # 100 for test, 2000 for real runs
num_chains = 4
target_accept_prob = 0.65

print("MCMC settings:")
print(f"  Warmup: {num_warmup}")
print(f"  Samples: {num_samples}")
print(f"  Chains: {num_chains}")
print(f"  Target accept prob: {target_accept_prob}")


# In[13]:


# Run MCMC
rng_mcmc = jr.key(456)

nuts_kernel = NUTS(
    model.model,
    target_accept_prob=target_accept_prob,
    init_strategy=numpyro.infer.init_to_value(values=init_params),
)

mcmc = MCMC(
    nuts_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method="parallel",
    progress_bar=False,
)

mcmc.run(rng_mcmc)

samples = mcmc.get_samples(group_by_chain=True)


# ## 4. Analyze Results
#
# Check convergence diagnostics and compare recovered parameters to truth.

# In[14]:


# Print MCMC diagnostics
mcmc.print_summary()

# Get divergences
divergences = mcmc.get_extra_fields(group_by_chain=True)["diverging"]
n_divergences = np.sum(divergences)
print(f"\nTotal divergences: {n_divergences}")


# In[15]:


# Compare to truth
param_names = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]

print("\nParameter recovery:")
print(f"{'Parameter':<10} {'Truth':<10} {'Mean':<10} {'Std':<10} {'Bias (σ)':<10}")
print("-" * 60)

for param in param_names:
    if param in samples:
        chain_samples = samples[param].reshape(-1)  # Flatten chains
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
fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 2 * len(param_names)))

for i, param in enumerate(param_names):
    if param in samples:
        ax = axes[i]
        chain_data = samples[param]

        # Plot each chain
        for chain_idx in range(num_chains):
            ax.plot(chain_data[chain_idx, :], alpha=0.7, label=f"Chain {chain_idx}")

        # Truth line
        ax.axhline(truth_params[param], color="red", linestyle="--", linewidth=2, label="Truth")

        ax.set_ylabel(param)
        ax.set_xlabel("Iteration")
        if i == 0:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[17]:


# Posterior distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, param in enumerate(param_names):
    if param in samples:
        ax = axes[i]
        chain_samples = samples[param].reshape(-1)

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
plt.show()


# ## 6. Corner Plot
#
# Show joint posterior distributions.

# In[18]:


try:
    import corner

    # Prepare data for corner plot
    corner_data = np.column_stack([samples[p].reshape(-1) for p in param_names if p in samples])
    corner_labels = [p for p in param_names if p in samples]
    corner_truths = [truth_params[p] for p in corner_labels]

    fig = corner.corner(
        corner_data,
        labels=corner_labels,
        truths=corner_truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.show()

except ImportError:
    print("Corner plot requires the 'corner' package. Install with: pip install corner")


# ## Summary
#
# This notebook demonstrated:
# - Creating a field-level model with specified configuration
# - Generating synthetic observations from truth parameters
# - Running MCMC inference with NUTS sampler
# - Analyzing convergence and parameter recovery
# - Visualizing chains and posteriors
