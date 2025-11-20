#!/usr/bin/env python
"""
Field-Level Inference on Synthetic Data

This script follows benchmark-field-level (MCLMC section):
https://github.com/hsimonfroy/benchmark-field-level

Improvements:
- YAML configuration
- Parallelized with jit(vmap) for multi-GPU
- Automated outputs and visualizations
"""

import argparse
import os
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
from jax import jit, tree, vmap

from desi_cmb_fli.model import FieldLevelModel, default_config
from desi_cmb_fli.samplers import get_mclmc_run, get_mclmc_warmup

jax.config.update("jax_enable_x64", True)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/04_field_level_inference/config.yaml")
args = parser.parse_args()

# Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Create output dirs in SCRATCH (not home)

scratch_dir = os.environ.get("SCRATCH", "/pscratch/sd/j/jhawla")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(scratch_dir) / "outputs" / f"run_{timestamp}"
fig_dir = run_dir / "figures"
config_dir = run_dir / "config"
for d in [fig_dir, config_dir]:
    d.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config, config_dir / "config.yaml")

print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Run dir: {run_dir}")

devices = jax.devices()
print(f"\nDevices: {len(devices)}")
for i, d in enumerate(devices):
    print(f"  {i}: {d}")

# Model setup
print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

model_config = default_config.copy()
model_config["mesh_shape"] = tuple(cfg["model"]["mesh_shape"])
model_config["box_shape"] = tuple(cfg["model"]["box_shape"])
model_config["evolution"] = cfg["model"]["evolution"]
model_config["lpt_order"] = cfg["model"]["lpt_order"]
model_config["a_obs"] = cfg["model"]["a_obs"]
model_config["gxy_density"] = cfg["model"]["gxy_density"]

model = FieldLevelModel(**model_config)
print(model)
model.save(config_dir / "model.yaml")

# Generate truth
print("\n" + "=" * 80)
print("GENERATING TRUTH")
print("=" * 80)

truth_params = cfg["truth_params"]
seed = cfg["seed"]
print(f"Truth params: {truth_params}")
print(f"Seed: {seed}")

truth = model.predict(
    samples=truth_params,
    hide_base=False,
    hide_samp=False,
    frombase=True,
    rng=jr.key(seed),
)
jnp.savez(config_dir / "truth.npz", **truth)

print(f"\nObs shape: {truth['obs'].shape}")
print(f"Mean count: {float(jnp.mean(truth['obs'])):.4f}")
print(f"Std: {float(jnp.std(truth['obs'])):.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
idx = model_config["mesh_shape"][0] // 2
slices = [truth["obs"][idx, :, :], truth["obs"][:, idx, :], truth["obs"][:, :, idx]]
titles = [f"YZ (x={idx})", f"XZ (y={idx})", f"XY (z={idx})"]
for ax, data, title in zip(axes, slices, titles, strict=True):
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Galaxy count")
plt.tight_layout()
plt.savefig(fig_dir / "obs_slices.png", dpi=150)
plt.close()

# MCMC config
print("\n" + "=" * 80)
print("MCMC CONFIGURATION")
print("=" * 80)

num_warmup = cfg["mcmc"]["num_warmup"]
num_samples = cfg["mcmc"]["num_samples"]
num_chains = cfg["mcmc"]["num_chains"]

mclmc_cfg = cfg["mcmc"].get("mclmc", {})
desired_energy_var = float(mclmc_cfg.get("desired_energy_var", 5e-4))
diagonal_precond = bool(mclmc_cfg.get("diagonal_preconditioning", False))
thinning = int(mclmc_cfg.get("thinning", 1))

print("Sampler: MCLMC")
print(f"Warmup: {num_warmup}, Samples: {num_samples}, Chains: {num_chains}")
print(f"Energy var: {desired_energy_var}, Diag precond: {diagonal_precond}, Thin: {thinning}")

if len(devices) >= num_chains:
    print(f"\n✓ {num_chains} chains // across {len(devices)} devices")
else:
    print(f"\n⚠️  {len(devices)} devices for {num_chains} chains")

# STEP 1: Warmup mesh only (benchmark approach)
print("\n" + "=" * 80)
print("STEP 1: WARMUP MESH ONLY")
print("=" * 80)

model.reset()
model.condition({"obs": truth["obs"]} | model.loc_fid, frombase=True)
model.block()

init_params_ = jit(vmap(partial(model.kaiser_post, delta_obs=truth["obs"] - 1)))(
    jr.split(jr.key(45), num_chains)
)
init_mesh_ = {k: init_params_[k] for k in ["init_mesh_"]}

print("Warming mesh (2^10 steps, cosmo/bias fixed)...")
warmup_mesh_fn = jit(
    vmap(
        get_mclmc_warmup(
            model.logpdf,
            n_steps=2**10,
            config=None,
            desired_energy_var=1e-6,
            diagonal_preconditioning=False,
        )
    )
)

state_mesh, config_mesh = warmup_mesh_fn(jr.split(jr.key(43), num_chains), init_mesh_)

print("✓ Mesh warmup done")
print(f"  Logdens (median): {float(jnp.median(state_mesh.logdensity)):.2f}")
print(f"  L (median): {float(jnp.median(config_mesh.L)):.6f}")
print(f"  step_size (median): {float(jnp.median(config_mesh.step_size)):.6f}")

init_params_ |= state_mesh.position

# STEP 2: Warmup all params
print("\n" + "=" * 80)
print("STEP 2: WARMUP ALL PARAMS")
print("=" * 80)

model.reset()
model.condition({"obs": truth["obs"]})
model.block()

print(f"Warming all params ({num_warmup} steps)...")
warmup_all_fn = jit(
    vmap(
        get_mclmc_warmup(
            model.logpdf,
            n_steps=num_warmup,
            config=None,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_precond,
        )
    )
)

state, config = warmup_all_fn(jr.split(jr.key(43), num_chains), init_params_)

print("✓ Full warmup done")
print(f"  Logdens (median): {float(jnp.median(state.logdensity)):.2f}")
median_L_adapted = float(jnp.median(config.L))
print(f"  Adapted L (median): {median_L_adapted:.6f}")
median_ss = float(jnp.median(config.step_size))
print(f"  step_size (median): {median_ss:.6f}")

# Recalculate L (benchmark approach)
eval_per_ess = 1e3
median_imm = jnp.median(config.inverse_mass_matrix, axis=0)
recalc_L = float(0.4 * eval_per_ess / 2 * median_ss)

config = MCLMCAdaptationState(L=recalc_L, step_size=float(median_ss), inverse_mass_matrix=median_imm)
config = tree.map(lambda x: jnp.broadcast_to(x, (num_chains, *jnp.shape(x))), config)

print(f"\n  Recalculated L: {recalc_L:.6f} (was: {median_L_adapted:.6f})")

jnp.savez(config_dir / "warmup_state.npz", **state.position)
with open(config_dir / "warmup_config.yaml", "w") as f:
    yaml.dump(
        {"L": recalc_L, "step_size": median_ss, "eval_per_ess": eval_per_ess},
        f,
    )

# STEP 3: Sample
print("\n" + "=" * 80)
print("STEP 3: SAMPLING")
print("=" * 80)

run_fn = jit(
    vmap(get_mclmc_run(model.logpdf, n_samples=num_samples, thinning=thinning, progress_bar=False))
)

print(f"\nRunning {num_chains} chains // for {num_samples} steps...")
key = jr.key(42)
key, run_key = jr.split(key)
state, samples_dict = run_fn(jr.split(run_key, num_chains), state, config)

print("\n✓ Sampling done!")

samples = {}
param_names = [k for k in samples_dict.keys() if k not in ["logdensity", "mse_per_dim"]]
for p in param_names:
    samples[p] = np.array(samples_dict[p])  # Convert JAX to NumPy for analysis

jnp.savez(config_dir / "samples.npz", **samples_dict)
print(f"\nSamples saved: {config_dir / 'samples.npz'}")
print(f"  Mean MSE/dim: {float(jnp.mean(samples_dict['mse_per_dim'])):.6e}")
print(f"  Final logdensity (mean): {float(jnp.mean(state.logdensity)):.2f}")

# Analysis
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

for p in param_names:
    vals = samples[p].reshape(-1)
    print(f"\n{p}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")

print("\n" + "=" * 80)
print("PARAMETER RECOVERY")
print("=" * 80)

comp_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
print(f"\n{'Param':<10} {'Truth':<10} {'Mean':<10} {'Std':<10} {'Bias(σ)':<10}")
print("-" * 60)

for p in comp_params:
    pk = p + "_"
    if pk in samples:
        vals = samples[pk].reshape(-1)
        mean, std = np.mean(vals), np.std(vals)
        truth_val = truth_params[p]
        bias = (mean - truth_val) / std if std > 0 else 0
        print(f"{p:<10} {truth_val:<10.4f} {mean:<10.4f} {std:<10.4f} {bias:<10.2f}")

# Viz: Traces
print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(len(comp_params), 1, figsize=(12, 2 * len(comp_params)))
for i, p in enumerate(comp_params):
    pk = p + "_"
    if pk in samples:
        ax = axes[i]
        for c in range(num_chains):
            ax.plot(samples[pk][c, :], alpha=0.7, lw=0.5, label=f"C{c}" if i == 0 else "")
        ax.axhline(truth_params[p], color="red", ls="--", lw=2, label="Truth" if i == 0 else "")
        ax.set_ylabel(p)
        ax.set_xlabel("Iter")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "traces.png", dpi=150)
plt.close()
print(f"✓ Traces: {fig_dir / 'traces.png'}")

# Viz: Posteriors
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, p in enumerate(comp_params):
    pk = p + "_"
    if pk in samples:
        ax = axes[i]
        vals = samples[pk].reshape(-1)
        ax.hist(vals, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")
        ax.axvline(truth_params[p], color="red", ls="--", lw=2, label="Truth")
        ax.axvline(np.mean(vals), color="green", ls="-", lw=2, label="Mean")
        ax.set_xlabel(p)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "posteriors.png", dpi=150)
plt.close()
print(f"✓ Posteriors: {fig_dir / 'posteriors.png'}")

# Viz: Corner
try:
    import corner

    labels, data_list, truths = [], [], []
    for p in comp_params:
        pk = p + "_"
        if pk in samples:
            labels.append(p)
            data_list.append(samples[pk].reshape(-1))
            truths.append(truth_params[p])
    data = np.column_stack(data_list)
    fig = corner.corner(
        data,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.savefig(fig_dir / "corner.png", dpi=150)
    plt.close()
    print(f"✓ Corner: {fig_dir / 'corner.png'}")
except ImportError:
    print("⚠️  Corner skipped (install 'corner')")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nOutputs: {run_dir}")
print(f"  Config: {config_dir}")
print(f"  Figures: {fig_dir}")
print(f"  Samples: {config_dir / 'samples.npz'}")
print(f"  Truth: {config_dir / 'truth.npz'}")
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
