#!/usr/bin/env python
"""
Joint Field-Level Inference: Galaxies + CMB Lensing

This script extends 04_field_level_inference.py to include CMB lensing convergence
as additional observational constraint for joint inference.

When cmb_lensing.enabled=false: behaves exactly like script 04
When cmb_lensing.enabled=true: uses joint likelihood p(obs_gal, kappa_obs | δ, cosmo, bias)
"""

import argparse
import gc
import os
import shutil

# Memory optimization for JAX on GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import yaml
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from jax import jit, pmap, tree

from desi_cmb_fli.chains import Chains
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

# Create output dirs in SCRATCH

scratch_dir = os.environ.get("SCRATCH", "/pscratch/sd/j/jhawla")
start_time = datetime.now()
timestamp = start_time.strftime("%Y%m%d_%H%M%S")
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

# CMB lensing config
cmb_cfg = cfg.get("cmb_lensing", {})
cmb_enabled = cmb_cfg.get("enabled", False)

if cmb_enabled:
    print("\n✓ CMB lensing ENABLED (joint inference)")
    model_config["cmb_enabled"] = True  # Enable CMB in model
    model_config["cmb_field_size_deg"] = float(cmb_cfg.get("field_size_deg", 5.0))
    model_config["cmb_field_npix"] = int(cmb_cfg.get("field_npix", 64))
    model_config["cmb_z_source"] = float(cmb_cfg.get("z_source", 1100.0))
    model_config["cmb_noise_std"] = float(cmb_cfg.get("noise_std", 0.01))
    print(f"  Field: {model_config['cmb_field_size_deg']}° × {model_config['cmb_field_npix']}px")
    print(f"  z_source: {model_config['cmb_z_source']}")
    print(f"  Noise σ: {model_config['cmb_noise_std']}")

    print(f"  Noise σ: {model_config['cmb_noise_std']}")

    # --- Validation: Lensing Kernel Coverage ---
    print("\nComputing lensing efficiency...")
    from desi_cmb_fli.bricks import get_cosmology
    from desi_cmb_fli.cmb_lensing import compute_lensing_efficiency

    # Create a temporary cosmology object for calculation
    # We use truth params or defaults if not available
    truth_params_val = cfg.get("truth_params", {})
    cosmo_val = get_cosmology(
        Omega_m=truth_params_val.get("Omega_m", 0.3),
        sigma8=truth_params_val.get("sigma8", 0.8),
        b1=1.0, b2=0.0, bs2=0.0, bn2=0.0 # Bias doesn't matter for distance
    )

    # Calculate box center chi based on a_obs
    import jax.numpy as jnp
    import jax_cosmo as jc
    chi_center_val = jc.background.radial_comoving_distance(cosmo_val, jnp.atleast_1d(model_config["a_obs"]))[0]
    chi_center_val = float(chi_center_val)
    box_size_z = model_config["box_shape"][2]

    z_grid, kernel, fraction, box_mask = compute_lensing_efficiency(
        cosmo_val,
        model_config["cmb_z_source"],
        chi_center_val,
        box_size_z
    )

    print(f"  Box center: z ~ {1.0/model_config['a_obs'] - 1.0:.2f} (chi = {chi_center_val:.1f} Mpc/h)")
    print(f"  Box coverage: {fraction*100:.2f}% of the lensing kernel")

    # Plot kernel and box coverage
    plt.figure(figsize=(8, 5))
    # Use semilogy for better visibility of the kernel structure
    plt.semilogy(z_grid, kernel, label=r"$W^\kappa(z)$")
    plt.fill_between(z_grid, 1e-10, kernel, where=box_mask, alpha=0.3, color="green", label=f"Box ({fraction:.1%})")
    plt.xlabel("Redshift z")
    plt.ylabel(r"$W^\kappa(z)$")
    plt.title(f"CMB Lensing Kernel Coverage (Box center z={1/model_config['a_obs']-1:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both", ls="-")
    plt.ylim(bottom=1e-5 * np.max(kernel)) # Set reasonable bottom limit
    plt.savefig(fig_dir / "lensing_kernel_coverage.png", dpi=150)
    plt.close()
    print(f"✓ Lensing coverage plot: {fig_dir / 'lensing_kernel_coverage.png'}")
    # -------------------------------------------

else:
    print("\n⚠️  CMB lensing DISABLED (galaxies only)")
    model_config["cmb_enabled"] = False

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
    hide_det=False,  # Keep deterministic sites for kappa_pred
    frombase=True,
    rng=jr.key(seed),
)
jnp.savez(config_dir / "truth.npz", **truth)

print(f"\nGalaxy obs shape: {truth['obs'].shape}")
print(f"Mean count: {float(jnp.mean(truth['obs'])):.4f}")
print(f"Std: {float(jnp.std(truth['obs'])):.4f}")

if cmb_enabled and "kappa_obs" in truth:
    print(f"\nCMB κ obs shape: {truth['kappa_obs'].shape}")
    print(f"Mean κ: {float(jnp.mean(truth['kappa_obs'])):.6f}")
    print(f"Std κ: {float(jnp.std(truth['kappa_obs'])):.6f}")
    if "kappa_pred" in truth:
        print(f"Mean κ_pred: {float(jnp.mean(truth['kappa_pred'])):.6f}")

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

# Visualize CMB convergence
if cmb_enabled and "kappa_obs" in truth:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Observed kappa
    im0 = axes[0].imshow(truth["kappa_obs"], origin="lower", cmap="RdBu_r")
    axes[0].set_title("CMB Convergence κ (observed)")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    plt.colorbar(im0, ax=axes[0], label="κ")

    # Predicted kappa (if available)
    if "kappa_pred" in truth:
        im1 = axes[1].imshow(truth["kappa_pred"], origin="lower", cmap="RdBu_r")
        axes[1].set_title("CMB Convergence κ (predicted)")
        axes[1].set_xlabel("x [pix]")
        axes[1].set_ylabel("y [pix]")
        plt.colorbar(im1, ax=axes[1], label="κ")

        # Residual
        residual = truth["kappa_obs"] - truth["kappa_pred"]
        im2 = axes[2].imshow(residual, origin="lower", cmap="RdBu_r")
        axes[2].set_title(f"Residual (σ={float(jnp.std(residual)):.6f})")
        axes[2].set_xlabel("x [pix]")
        axes[2].set_ylabel("y [pix]")
        plt.colorbar(im2, ax=axes[2], label="κ_obs - κ_pred")

    plt.tight_layout()
    plt.savefig(fig_dir / "cmb_kappa_truth.png", dpi=150)
    plt.close()
    print(f"✓ CMB visualization: {fig_dir / 'cmb_kappa_truth.png'}")

    # Cross-correlation validation (CMB-Galaxy)
    print("\nComputing CMB-Galaxy cross-correlation...")
    from scipy.ndimage import map_coordinates

    # Project galaxy field onto CMB angular coordinates (same as convergence_Born)
    field_size_deg = model_config["cmb_field_size_deg"]
    field_npix = model_config["cmb_field_npix"]
    box_size = model_config["box_shape"][0]
    mesh_shape_val = model_config["mesh_shape"][0]

    # Angular coordinate grid
    theta = np.linspace(-field_size_deg/2, field_size_deg/2, field_npix, endpoint=False)
    rowgrid, colgrid = np.meshgrid(theta, theta, indexing='ij')
    coords = np.array(np.stack([rowgrid, colgrid], axis=0)) * (np.pi / 180.0)

    # Distance planes
    dx = box_size / mesh_shape_val
    dz = box_size / mesh_shape_val

    # Recalculate box center for validation (same as in model)
    # Use the same cosmology as truth generation
    chi_center_val = jc.background.radial_comoving_distance(cosmo_val, jnp.atleast_1d(model_config["a_obs"]))[0]
    chi_center_val = float(chi_center_val)

    r_start = chi_center_val - box_size / 2.0
    r_planes = r_start + (np.arange(mesh_shape_val) + 0.5) * dz

    # Project galaxy field (SUM along LOS, like kappa integral)
    gxy_field = np.array(truth['obs'])
    gxy_proj = np.zeros((field_npix, field_npix))

    for i in range(mesh_shape_val):
        r = r_planes[i]
        pixel_coords = coords * r / dx + mesh_shape_val / 2.0
        plane = map_coordinates(gxy_field[:, :, i], pixel_coords, order=1, mode='wrap')
        gxy_proj += plane

    # Compute cross-correlation
    kappa_flat = np.array(truth['kappa_obs']).flatten()
    gxy_proj_flat = gxy_proj.flatten()
    corr_cross = np.corrcoef(kappa_flat, gxy_proj_flat)[0, 1]

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(truth['kappa_obs'], origin='lower', cmap='RdBu_r')
    axes[0].set_title('CMB Convergence κ')
    axes[0].set_xlabel('x [pix]')
    axes[0].set_ylabel('y [pix]')
    plt.colorbar(im0, ax=axes[0], label='κ')

    im1 = axes[1].imshow(gxy_proj, origin='lower', cmap='viridis')
    axes[1].set_title('Galaxy Field ⟨count⟩ (projected)')
    axes[1].set_xlabel('x [pix]')
    axes[1].set_ylabel('y [pix]')
    plt.colorbar(im1, ax=axes[1], label='count')

    axes[2].scatter(gxy_proj_flat, kappa_flat, alpha=0.3, s=5, c='blue', edgecolors='none')
    axes[2].set_xlabel('Galaxy count (projected)')
    axes[2].set_ylabel('CMB Convergence κ')
    axes[2].set_title('Cross-Correlation: CMB vs Galaxies')
    axes[2].grid(True, alpha=0.3)

    color = 'green' if corr_cross > 0.8 else ('orange' if corr_cross > 0.5 else 'red')
    axes[2].text(0.05, 0.95, f"r = {corr_cross:.4f}",
                transform=axes[2].transAxes,
                bbox={'boxstyle': 'round', 'facecolor': color, 'alpha': 0.8, 'edgecolor': 'black'},
                verticalalignment='top', fontsize=14, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(fig_dir / "cmb_galaxy_cross_correlation.png", dpi=150)
    plt.close()

    print(f"✓ Cross-correlation: {fig_dir / 'cmb_galaxy_cross_correlation.png'}")
    print(f"  κ vs galaxy field: r = {corr_cross:.4f}")
    print(f"  Field size: {field_size_deg}°, npix: {field_npix}")
    print(f"  Galaxy projection: mean={gxy_proj.mean():.4f}, std={gxy_proj.std():.4f}")
    print(f"  CMB κ: mean={np.array(truth['kappa_obs']).mean():.6f}, std={np.array(truth['kappa_obs']).std():.6f}")

# MCMC config
print("\n" + "=" * 80)
print("MCMC CONFIGURATION")
print("=" * 80)

num_warmup = cfg["mcmc"]["num_warmup"]
num_samples = cfg["mcmc"]["num_samples"]  # Samples PER BATCH
num_batches = cfg["mcmc"].get("num_batches", 1)  # Number of batches
num_chains = cfg["mcmc"]["num_chains"]

mclmc_cfg = cfg["mcmc"].get("mclmc", {})
desired_energy_var = float(mclmc_cfg.get("desired_energy_var", 5e-4))
diagonal_precond = bool(mclmc_cfg.get("diagonal_preconditioning", False))

print("Sampler: MCLMC (Multi-chain + Mini-batch)")
print(f"Chains: {num_chains}")
print(f"Warmup: {num_warmup}")
print(f"Samples per batch: {num_samples}")
print(f"Number of batches: {num_batches}")
print(f"Total samples per chain: {num_samples * num_batches}")
print(f"Total samples (all chains): {num_samples * num_batches * num_chains}")
print(f"Energy var: {desired_energy_var}, Diag precond: {diagonal_precond}")

if len(devices) >= num_chains:
    print(f"\n✓ {num_chains} chains // across {len(devices)} GPUs")
    if len(devices) > num_chains:
        print(f"  (Note: {len(devices) - num_chains} GPUs idle)")
else:
    print(f"\n⚠️  WARNING: Only {len(devices)} GPUs available for {num_chains} chains.")
    print("   Some chains will share GPUs, which may slow down sampling significantly.")

# Condition model on observations
condition_dict = {"obs": truth["obs"]}
if cmb_enabled and "kappa_obs" in truth:
    condition_dict["kappa_obs"] = truth["kappa_obs"]
    print("\n✓ Model conditioned on galaxy obs + CMB κ obs")
else:
    print("\n✓ Model conditioned on galaxy obs only")

# STEP 1: Warmup mesh only (benchmark approach)
print("\n" + "=" * 80)
print("STEP 1: WARMUP MESH ONLY")
print("=" * 80)

model.reset()
model.condition(condition_dict | model.loc_fid, frombase=True)
model.block()

# Initialize (multi-chains)
delta_obs = truth["obs"] - 1
rngs = jr.split(jr.key(45), num_chains)

# Define a wrapper to handle the method call cleanly with pmap
def init_fn(rng, delta):
    return model.kaiser_post(rng, delta)

init_params_ = pmap(jit(init_fn), in_axes=(0, None))(rngs, delta_obs)
init_mesh_ = {k: init_params_[k] for k in ["init_mesh_"]}

print("Warming mesh (2^10 steps, cosmo/bias fixed)...")
warmup_mesh_fn = pmap(jit(get_mclmc_warmup(
    model.logpdf,
    n_steps=2**10,
    config=None,
    desired_energy_var=1e-6,
    diagonal_preconditioning=False,
)))

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
model.condition(condition_dict)
model.block()

print(f"Warming all params ({num_warmup} steps)...")
warmup_all_fn = pmap(jit(get_mclmc_warmup(
    model.logpdf,
    n_steps=num_warmup,
    config=None,
    desired_energy_var=desired_energy_var,
    diagonal_preconditioning=diagonal_precond,
)))

state, config = warmup_all_fn(jr.split(jr.key(43), num_chains), init_params_)

print("✓ Full warmup done")
median_L_adapted = float(jnp.median(config.L))
median_ss = float(jnp.median(config.step_size))
median_imm = jnp.median(config.inverse_mass_matrix, axis=0)
print(f"  Logdens (median): {float(jnp.median(state.logdensity)):.2f}")
print(f"  Adapted L (median): {median_L_adapted:.6f}")
print(f"  step_size (median): {median_ss:.6f}")

# Recalculate L (benchmark approach)
eval_per_ess = 1e3
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

# STEP 3: Multi-Chain Mini-Batch Sampling
print("\n" + "=" * 80)
print("STEP 3: MULTI-CHAIN MINI-BATCH SAMPLING")
print("=" * 80)

# Setup sampling function (pmap for parallel chains)
run_fn = pmap(jit(get_mclmc_run(model.logpdf, n_samples=num_samples, thinning=1, progress_bar=False)))

print(f"\nRunning {num_chains} chains in parallel, each with {num_batches} sequential batches")
print(f"   Samples per batch: {num_samples}")
print(f"   Total per chain: {num_samples * num_batches}")
print(f"   Total all chains: {num_samples * num_batches * num_chains}\n")



# Storage
samples_scalars = {}  # For small parameters (scalars)
param_names = None

# Optional: control saving of large fields
mcmc_io_cfg = cfg.get("mcmc", {})
save_large_fields = bool(mcmc_io_cfg.get("save_large_fields", False))

key = jr.key(42)

for batch_idx in range(num_batches):
    print(f"Batch {batch_idx + 1}/{num_batches}:")

    # Run sampling for this batch (parallel across chains)
    key, *run_keys = jr.split(key, num_chains + 1)
    run_keys = jnp.array(run_keys)
    state, samples_dict = run_fn(run_keys, state, config)

    # Identify parameters on first batch
    if param_names is None:
        param_names = [k for k in samples_dict.keys() if k not in ["logdensity", "mse_per_dim"]]
        # Identify large fields
        large_params = {
            "init_mesh_", "init_mesh",
            "gxy_mesh", "matter_mesh",
            "lpt_pos", "rsd_pos", "nbody_pos",
            "obs"
        }
        scalar_params = [p for p in param_names if p not in large_params]

        # Initialize storage for scalars
        for p in scalar_params:
            samples_scalars[p] = []

    # Process batch
    batch_scalars = {}
    batch_large = {}

    for p in param_names:
        val = jax.device_get(samples_dict[p]) # Move to CPU
        if p in scalar_params:
            samples_scalars[p].append(val)
            batch_scalars[p] = val
        elif save_large_fields:
            batch_large[p] = val

    # Save batch to disk
    # We save everything for this batch in one file (or split if needed, but simple is better)
    batch_content = {**batch_scalars, **batch_large}
    jnp.savez(config_dir / f"samples_batch_{batch_idx}.npz", **batch_content)

    print(f"  ✓ {num_samples} samples × {num_chains} chains")
    print(f"  MSE/dim (median): {float(jnp.median(samples_dict['mse_per_dim'])):.6e}")
    print(f"  Logdensity (median): {float(jnp.median(state.logdensity)):.2f}")
    print(f"  Saved: samples_batch_{batch_idx}.npz")

    # Explicitly delete large arrays to free memory
    del batch_large, batch_content, batch_scalars
    gc.collect()

print("\n✓ All batches complete!")

# Aggregate scalar samples
print("\nAggregating scalar samples...")
samples = {}
for p, vals in samples_scalars.items():
    # vals is list of (num_chains, num_samples) arrays
    # Concatenate along sample axis (axis 1)
    samples[p] = np.concatenate(vals, axis=1)
    print(f"  {p}: {samples[p].shape}")

# Save combined scalars
print(f"\nSaving combined samples (scalars only): {config_dir / 'samples.npz'}")
jnp.savez(config_dir / "samples.npz", **samples)
first_key = next(iter(samples.keys()))
print(f"  Total samples: {samples[first_key].shape}  # (chains={num_chains}, samples={num_samples * num_batches})")

# Analysis
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS (R-hat)")
print("=" * 80)

# REPARAMETRIZATION TO PHYSICAL SPACE
print("\nReparametrizing samples to physical space for analysis...")
# Exclude large fields to avoid OOM during reparam
large_fields = ["init_mesh_", "gxy_mesh", "matter_mesh", "lpt_pos", "rsd_pos", "nbody_pos", "obs"]
samples_jax = {
    k: jnp.array(v)
    for k, v in samples.items()
    if k.endswith('_') and k not in large_fields
}

# Create Chains object for reparam
chain_obj = Chains(samples_jax, model.groups | model.groups_)
physical_chains = model.reparam_chains(chain_obj, fourier=False, batch_ndim=2)
physical_samples = {k: np.array(v) for k, v in physical_chains.data.items()}

print("Physical keys:", physical_samples.keys())

# Use physical samples for stats and plots
# Update param_names to physical names
comp_params = ["Omega_m", "sigma8", "b1", "b2", "bs2", "bn2"]
# Note: physical_samples keys do NOT have underscores (e.g. "Omega_m")

def compute_rhat(chains, burnin_frac=0.5):
    """Compute Gelman-Rubin R-hat for convergence.

    chains: (num_chains, num_samples)
    burnin_frac: fraction of samples to discard as burn-in (default 0.5)
    """
    num_chains, num_samples = chains.shape

    # Discard burn-in (standard: 50% of post-warmup samples)
    burnin = int(num_samples * burnin_frac)
    chains = chains[:, burnin:]
    _, num_samples = chains.shape

    # Chain means (after burn-in)
    chain_means = np.mean(chains, axis=1)
    # Grand mean
    grand_mean = np.mean(chain_means)

    # R-hat
    if num_chains < 2:
        return float('nan')

    # Between-chain variance
    B = num_samples / (num_chains - 1) * np.sum((chain_means - grand_mean)**2)

    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)

    # Pooled variance estimate
    var_plus = ((num_samples - 1) / num_samples) * W + (1 / num_samples) * B

    # R-hat
    rhat = np.sqrt(var_plus / W) if W > 0 else np.inf
    return rhat

print(f"\n{'Param':<10} {'R-hat':<10} {'Status':<15}")
print("-" * 40)
print("(After discarding 50% burn-in)")

for p in comp_params:
    # Use physical parameter name directly (no underscore)
    if p in physical_samples:
        rhat = compute_rhat(physical_samples[p], burnin_frac=0.5)
        status = "✓ Converged" if rhat < 1.1 else ("⚠️  Marginal" if rhat < 1.2 else "❌ Not converged")
        print(f"{p:<10} {rhat:<10.4f} {status:<15}")

print("\n" + "=" * 80)
print("STATISTICS (All Chains Combined)")
print("=" * 80)

for p in comp_params:
    if p in physical_samples:
        vals = physical_samples[p].reshape(-1)  # Flatten (chains, samples) → 1D
        print(f"\n{p}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")

print("\n" + "=" * 80)
print("PARAMETER RECOVERY")
print("=" * 80)

print(f"\n{'Param':<10} {'Truth':<10} {'Mean':<10} {'Std':<10} {'Bias(σ)':<10}")
print("-" * 60)

for p in comp_params:
    if p in physical_samples:
        # Pool all chains (R-hat validated convergence, so treat as single posterior)
        vals = physical_samples[p].reshape(-1)  # Flatten (chains, samples) → 1D
        mean, std = np.mean(vals), np.std(vals)
        truth_val = truth_params.get(p, float('nan'))
        bias = (mean - truth_val) / std if std > 0 else 0
        print(f"{p:<10} {truth_val:<10.4f} {mean:<10.4f} {std:<10.4f} {bias:<10.2f}")

# Viz: Traces
print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(len(comp_params), 1, figsize=(12, 2 * len(comp_params)))
for i, p in enumerate(comp_params):
    if p in physical_samples:
        ax = axes[i]
        # Plot each chain separately
        for c in range(num_chains):
            ax.plot(physical_samples[p][c, :], alpha=0.7, lw=0.5, label=f"Chain {c}" if i == 0 else "")
        if p in truth_params:
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
    if p in physical_samples:
        ax = axes[i]
        vals = physical_samples[p].reshape(-1)  # Flatten (chains, samples) → 1D
        ax.hist(vals, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")
        if p in truth_params:
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

# Viz: GetDist Triangle Plot
try:
    from getdist import MCSamples, plots

    print("\nGenerating GetDist triangle plot...")

    gd_samples = []

    # Check which params are available
    available_params = [p for p in comp_params if p in physical_samples]
    available_labels = list(available_params)
    available_truths = [truth_params.get(p, None) for p in available_params]

    if available_params:
        # Iterate over chains
        for c in range(num_chains):
            # Stack parameters for this chain: (nsamples, nparams)
            chain_data = []
            for p in available_params:
                chain_data.append(physical_samples[p][c, :])

            # Transpose to (nsamples, nparams)
            chain_data = np.column_stack(chain_data)
            gd_samples.append(chain_data)
            print(f"  Chain {c} shape: {chain_data.shape}")

        # Create MCSamples object
        # settings={'ignore_rows': 0.5} to discard first 50% as burn-in (consistent with R-hat check)
        samples_gd = MCSamples(samples=gd_samples, names=available_params, labels=available_labels,
                               settings={'ignore_rows': 0.5})

        # Plot
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples_gd], filled=True, title_limit=1)

        g.export(str(fig_dir / "corner.png"))
        print(f"✓ GetDist plot: {fig_dir / 'corner.png'}")

except ImportError:
    print("⚠️  GetDist skipped (install 'getdist')")
except Exception as e:
    print(f"⚠️  GetDist plotting failed: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nOutputs: {run_dir}")
print(f"  Config: {config_dir}")
print(f"  Figures: {fig_dir}")
print(f"  Samples: {config_dir / 'samples.npz'}")
print(f"  Truth: {config_dir / 'truth.npz'}")
if cmb_enabled:
    print("\n✓ Joint inference: Galaxies + CMB lensing")
else:
    print("\n✓ Galaxy-only inference")
print("\n" + "=" * 80)
end_time = datetime.now()
duration = end_time - start_time
print(f"Total run time: {duration}")
print("DONE")
print("=" * 80)
