#!/usr/bin/env python
"""
Field-Level Inference with CMB Lensing

Extension of 04_field_level_inference.py that demonstrates CMB lensing
convergence maps from galaxy clustering inference.

This script:
1. Runs field-level inference on galaxy clustering (same as 04)
2. Computes CMB lensing convergence κ(θ) from inferred density fields
3. Analyzes correlation between galaxy and CMB lensing observables
"""

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
from jax import jit, tree, vmap

from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.cmb_lensing import density_field_to_convergence
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

# Force CMB lensing enabled
if "cmb_lensing" not in cfg:
    cfg["cmb_lensing"] = {}
cfg["cmb_lensing"]["enabled"] = True
cfg["cmb_lensing"].setdefault("field_size_deg", 5.0)
cfg["cmb_lensing"].setdefault("field_npix", 64)
cfg["cmb_lensing"].setdefault("z_source", 1.0)

# Create output dirs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(f"outputs/run_{timestamp}_cmb")
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

# CMB config
cmb_cfg = cfg["cmb_lensing"]
print("\nCMB Lensing:")
print(f"  Field size: {cmb_cfg['field_size_deg']}°")
print(f"  Resolution: {cmb_cfg['field_npix']} pixels")
print(f"  Source z: {cmb_cfg['z_source']}")

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

print(f"\nGalaxy field shape: {truth['obs'].shape}")
print(f"Mean count: {float(jnp.mean(truth['obs'])):.4f}")
print(f"Std: {float(jnp.std(truth['obs'])):.4f}")

# Compute CMB lensing for truth
print("\nComputing CMB convergence...")
cosmo_truth = get_cosmology(Omega_m=truth_params["Omega_m"], sigma8=truth_params["sigma8"])
b1_truth = truth_params["b1"]
delta_matter_truth = (truth["gxy_mesh"] - 1.0) / b1_truth
density_field_truth = 1.0 + delta_matter_truth

kappa_truth = density_field_to_convergence(
    density_field_truth,
    model.box_shape,
    cosmo_truth,
    cmb_cfg["field_size_deg"],
    cmb_cfg["field_npix"],
    cmb_cfg["z_source"],
)

jnp.savez(config_dir / "kappa_truth.npz", kappa=kappa_truth)
print(f"Convergence shape: {kappa_truth.shape}")
print(f"Mean κ: {float(jnp.mean(kappa_truth)):.6e}")
print(f"Std κ: {float(jnp.std(kappa_truth)):.6e}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Galaxy field
idx = model_config["mesh_shape"][0] // 2
im0 = axes[0].imshow(truth["obs"][:, :, idx], origin="lower", cmap="viridis")
axes[0].set_title(f"Galaxy field (z={idx})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0], label="1+δ_gxy")

# Matter density
im1 = axes[1].imshow(density_field_truth[:, :, idx], origin="lower", cmap="RdBu_r")
axes[1].set_title(f"Matter density (z={idx})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im1, ax=axes[1], label="1+δ_m")

# CMB convergence
extent = [0, cmb_cfg["field_size_deg"], 0, cmb_cfg["field_size_deg"]]
im2 = axes[2].imshow(kappa_truth, origin="lower", cmap="RdBu_r", extent=extent)
axes[2].set_title(f"CMB Convergence (z_s={cmb_cfg['z_source']})")
axes[2].set_xlabel("θ_x [deg]")
axes[2].set_ylabel("θ_y [deg]")
plt.colorbar(im2, ax=axes[2], label="κ")

plt.tight_layout()
plt.savefig(fig_dir / "truth_comparison.png", dpi=150)
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

# STEP 1: Warmup mesh only
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

# Recalculate L
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

print(f"\n🚀 Running {num_chains} chains // for {num_samples} steps...")
key = jr.key(42)
key, run_key = jr.split(key)
state, samples_dict = run_fn(jr.split(run_key, num_chains), state, config)

print("\n✓ Sampling done!")

samples = {}
param_names = [k for k in samples_dict.keys() if k not in ["logdensity", "mse_per_dim"]]
for p in param_names:
    samples[p] = np.array(samples_dict[p])

jnp.savez(config_dir / "samples.npz", **samples_dict)
print(f"\nSamples saved: {config_dir / 'samples.npz'}")
print(f"  Mean MSE/dim: {float(jnp.mean(samples_dict['mse_per_dim'])):.6e}")
print(f"  Final logdensity (mean): {float(jnp.mean(state.logdensity)):.2f}")

# CMB Lensing from posterior samples
print("\n" + "=" * 80)
print("CMB LENSING FROM POSTERIOR")
print("=" * 80)

print("Computing κ from posterior samples...")

def compute_kappa_for_sample(sample_dict):
    """Compute kappa for a single posterior sample."""
    # Get cosmology
    cosmo = get_cosmology(Omega_m=sample_dict["Omega_m"], sigma8=sample_dict["sigma8"])

    # Get initial mesh and evolve to get gxy_mesh
    # For simplicity, we'll recompute the full forward model
    pred = model.predict(
        samples=sample_dict,
        hide_base=False,
        hide_det=False,
        frombase=False,
        rng=jr.key(0),  # Deterministic from init_mesh
    )

    # Convert to matter density
    b1 = sample_dict["b1"]
    delta_matter = (pred["gxy_mesh"] - 1.0) / b1
    density_field = 1.0 + delta_matter

    # Compute convergence
    kappa = density_field_to_convergence(
        density_field,
        model.box_shape,
        cosmo,
        cmb_cfg["field_size_deg"],
        cmb_cfg["field_npix"],
        cmb_cfg["z_source"],
    )

    return kappa

# Compute for subset of samples (computationally expensive)
n_kappa_samples = min(100, num_samples * num_chains // thinning)
print(f"Computing κ for {n_kappa_samples} samples...")

kappa_samples = []
sample_indices = np.linspace(0, num_samples - 1, n_kappa_samples, dtype=int)

for i, idx in enumerate(sample_indices):
    if i % 10 == 0:
        print(f"  Progress: {i}/{n_kappa_samples}")

    # Get sample from first chain
    sample_i = {k: samples[k][0, idx] for k in param_names}
    kappa_i = compute_kappa_for_sample(sample_i)
    kappa_samples.append(kappa_i)

kappa_samples = jnp.array(kappa_samples)
jnp.savez(config_dir / "kappa_samples.npz", kappa=kappa_samples)

print(f"\n✓ Computed {len(kappa_samples)} κ maps")
print(f"  Mean κ (posterior): {float(jnp.mean(kappa_samples)):.6e}")
print(f"  Std κ (posterior): {float(jnp.std(kappa_samples)):.6e}")

# Statistics
kappa_mean = jnp.mean(kappa_samples, axis=0)
kappa_std = jnp.std(kappa_samples, axis=0)

# Analysis
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

# Visualizations
print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

# 1. Parameter traces
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

# 2. CMB convergence comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Truth
im0 = axes[0].imshow(kappa_truth, origin="lower", cmap="RdBu_r", extent=extent)
axes[0].set_title("Truth κ")
axes[0].set_xlabel("θ_x [deg]")
axes[0].set_ylabel("θ_y [deg]")
plt.colorbar(im0, ax=axes[0], label="κ")

# Posterior mean
im1 = axes[1].imshow(kappa_mean, origin="lower", cmap="RdBu_r", extent=extent)
axes[1].set_title("Posterior Mean κ")
axes[1].set_xlabel("θ_x [deg]")
axes[1].set_ylabel("θ_y [deg]")
plt.colorbar(im1, ax=axes[1], label="κ")

# Posterior std
im2 = axes[2].imshow(kappa_std, origin="lower", cmap="viridis", extent=extent)
axes[2].set_title("Posterior Std κ")
axes[2].set_xlabel("θ_x [deg]")
axes[2].set_ylabel("θ_y [deg]")
plt.colorbar(im2, ax=axes[2], label="σ_κ")

plt.tight_layout()
plt.savefig(fig_dir / "kappa_posterior.png", dpi=150)
plt.close()

# 3. Residuals
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
residuals = (kappa_truth - kappa_mean) / kappa_std
im = ax.imshow(residuals, origin="lower", cmap="seismic", extent=extent, vmin=-3, vmax=3)
ax.set_title("Normalized Residuals: (Truth - Mean) / Std")
ax.set_xlabel("θ_x [deg]")
ax.set_ylabel("θ_y [deg]")
plt.colorbar(im, ax=ax, label="σ")
plt.savefig(fig_dir / "kappa_residuals.png", dpi=150)
plt.close()

print(f"✓ CMB visualizations: {fig_dir}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nOutputs: {run_dir}")
print(f"  Config: {config_dir}")
print(f"  Figures: {fig_dir}")
print(f"  Samples: {config_dir / 'samples.npz'}")
print(f"  Truth: {config_dir / 'truth.npz'}")
print(f"  Kappa truth: {config_dir / 'kappa_truth.npz'}")
print(f"  Kappa samples: {config_dir / 'kappa_samples.npz'}")
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
