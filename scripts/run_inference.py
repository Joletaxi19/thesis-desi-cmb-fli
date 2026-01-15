#!/usr/bin/env python
"""
Joint Field-Level Inference: Galaxies + CMB Lensing

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
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from jax import jit, pmap, tree

from desi_cmb_fli import cmb_lensing, metrics, plot, utils
from desi_cmb_fli.bricks import get_cosmology
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.model import FieldLevelModel, default_config
from desi_cmb_fli.samplers import get_mclmc_run, get_mclmc_warmup

jax.config.update("jax_enable_x64", True)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/inference/config.yaml")
args = parser.parse_args()

# Load config
cfg = utils.yload(args.config)

# Create output dirs in SCRATCH

scratch_dir = os.environ.get("SCRATCH", "/pscratch/sd/j/jhawla")
start_time = datetime.now()
timestamp = start_time.strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get("SLURM_JOB_ID", "local")
run_dir = Path(scratch_dir) / "outputs" / f"run_{timestamp}_{job_id}"
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
model_config["box_shape"] = tuple(cfg["model"]["box_shape"])

cell_size = float(cfg["model"]["cell_size"])
mesh_shape = [int(round(L / cell_size)) for L in model_config["box_shape"]]
# Ensure they are even
mesh_shape = [n + 1 if n % 2 != 0 else n for n in mesh_shape]
model_config["mesh_shape"] = tuple(mesh_shape)
print(f"Computed mesh_shape: {model_config['mesh_shape']} (from cell_size={cell_size})")

model_config["evolution"] = cfg["model"]["evolution"]
model_config["lpt_order"] = cfg["model"]["lpt_order"]

model_config["gxy_density"] = cfg["model"]["gxy_density"]

# CMB lensing config
cmb_cfg = cfg.get("cmb_lensing", {})
cmb_enabled = cmb_cfg.get("enabled", False)

if cmb_enabled:
    print("\n✓ CMB lensing ENABLED (joint inference)")
    model_config["cmb_enabled"] = True  # Enable CMB in model
    # Field size will be auto-calculated by the model
    model_config["cmb_field_size_deg"] = None
    model_config["cmb_field_npix"] = int(cmb_cfg["field_npix"]) if "field_npix" in cmb_cfg else None

    model_config["full_los_correction"] = cmb_cfg.get("full_los_correction", False)
    if model_config["full_los_correction"]:
        print("✓ full_los_correction ENABLED")

    model_config["cmb_z_source"] = float(cmb_cfg.get("z_source", 1100.0))

    if "cmb_noise_nell" in cmb_cfg:
        model_config["cmb_noise_nell"] = cmb_cfg["cmb_noise_nell"]
        print(f"  Noise: Using N_ell from {model_config['cmb_noise_nell']}")
    else:
        # Fallback or error if not provided
        raise ValueError("cmb_noise_nell must be provided in config when cmb_enabled=True")

    print(f"  z_source: {model_config['cmb_z_source']}")

else:
    print("\n⚠️  CMB lensing DISABLED (galaxies only)")
    model_config["cmb_enabled"] = False

model = FieldLevelModel(**model_config)
print(model)
model.save(config_dir / "model.yaml")

# Plot N_ell if CMB is enabled
if model.cmb_enabled and model.cmb_noise_nell is not None:
    print("\nPlotting CMB noise spectrum...")
    try:
        if isinstance(model.cmb_noise_nell, str):
            data = np.loadtxt(model.cmb_noise_nell)
            if data.ndim == 1:
                ell_in = np.arange(len(data))
                nell_in = data
            else:
                ell_in, nell_in = data[:, 0], data[:, 1]
        elif isinstance(model.cmb_noise_nell, dict):
            ell_in, nell_in = model.cmb_noise_nell["ell"], model.cmb_noise_nell["N_ell"]
        else:
            # Tuple or list
            ell_in, nell_in = model.cmb_noise_nell[0], model.cmb_noise_nell[1]

        # Filter for log plot
        mask = (ell_in > 0) & (nell_in > 0)

        plt.figure(figsize=(8, 6))
        plot.plot_pow(ell_in[mask], nell_in[mask], log=True, ylabel=r"$N_\ell$")
        plt.title("CMB Lensing Noise Power Spectrum")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(["Input $N_\\ell$"])
        plt.savefig(fig_dir / "cmb_noise_spectrum.png", dpi=150)
        plt.close()
        print(f"✓ CMB noise plot: {fig_dir / 'cmb_noise_spectrum.png'}")
    except Exception as e:
        print(f"⚠️  Could not plot N_ell: {e}")



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
    hide_det=False,
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
slices = [truth["obs"], truth["obs"], truth["obs"]]
slis = [
    (idx, 0),  # x=idx (last axis in plot_mesh, so axis=0) -> No, plot_mesh default axis=-1.
    # truth['obs'] is [x, y, z].
    # YZ plane (x fixed): axis=0
    # XZ plane (y fixed): axis=1
    # XY plane (z fixed): axis=2
]
# We use plot_mesh(mesh, sli=idx, axis=ax) to slice at index idx along axis ax
titles = [f"YZ (x={idx})", f"XZ (y={idx})", f"XY (z={idx})"]
axes_idx = [0, 1, 2]

for ax, axis_idx, title in zip(axes, axes_idx, titles, strict=True):
    plt.sca(ax)
    plot.plot_mesh(truth["obs"], sli=slice(idx, idx + 1), axis=axis_idx, cmap="viridis")
    ax.set_title(title)
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

    # Project galaxy field (SUM along LOS, like kappa integral)
    # Needed for C_l^gg and C_l^kg
    gxy_field = np.array(truth["obs"])

    # Recalculate box center for validation (same as in model)
    cosmo_val = get_cosmology(**truth_params)
    chi_center_val = jc.background.radial_comoving_distance(cosmo_val, jnp.atleast_1d(model_config["a_obs"]))[0]
    chi_center_val = float(chi_center_val)

    gxy_proj = cmb_lensing.project_flat_sky(
        gxy_field,
        model_config["box_shape"],
        model.cmb_field_size_deg,
        model.cmb_field_npix,
        chi_center_val
    )

    # Report stats
    print(f"  Field size: {model.cmb_field_size_deg}°, npix: {model.cmb_field_npix}")
    print(f"  Galaxy projection: mean={gxy_proj.mean():.4f}, std={gxy_proj.std():.4f}")

    # =========================================================================
    # DIAGNOSTIC SPECTRA: C_l
    # =========================================================================
    print("\nComputing Diagnostic Power Spectra C_l...")

    # 1. Reconstruct kappa_box (without high-z correction)
    print("  Reconstructing Box-Only Kappa...")
    kappa_box = cmb_lensing.density_field_to_convergence(
        truth['matter_mesh'],
        model.box_shape,
        cosmo_val,
        model.cmb_field_size_deg,
        model.cmb_field_npix,
        model.cmb_z_source,
        box_center_chi=chi_center_val,
    )
    if kappa_box.ndim == 3 and kappa_box.shape[0] == 1:
        kappa_box = kappa_box.squeeze(0)
    kappa_box = jnp.asarray(kappa_box, dtype=jnp.float64)  # Match dtype with kappa_pred

    # 2. Identify fields (Note: kappa_pred = box signal only. High-z goes into likelihood variance, not signal)
    kappa_total = truth['kappa_pred']
    kappa_obs = truth['kappa_obs']    # Signal + Noise

    # 3. Compute Spectra
    field_size = model.cmb_field_size_deg

    # Kappa Auto-correlations
    ell, cl_kk_box = metrics.get_cl_2d(kappa_box, field_size_deg=field_size)
    _, cl_kk_obs = metrics.get_cl_2d(kappa_obs, field_size_deg=field_size)

    # Galaxy Auto-correlation
    gxy_mean = jnp.mean(gxy_proj)
    gxy_delta = (gxy_proj - gxy_mean) / gxy_mean
    _, cl_gg = metrics.get_cl_2d(gxy_delta, field_size_deg=field_size)

    # Kappa-Galaxy Cross-correlation
    _, cl_kg = metrics.get_cl_2d(kappa_box, gxy_delta, field_size_deg=field_size)

    # 3.5 Compute Theoretical Spectra
    print("  Computing theoretical spectra (Limber)...")
    from desi_cmb_fli.cmb_lensing import (
        compute_theoretical_cl_gg,
        compute_theoretical_cl_kappa,
        compute_theoretical_cl_kg,
    )

    chi_min = 1.0  # Small positive value to avoid div by 0 in Limber
    chi_max = float(model.box_shape[2])
    z_source = model.cmb_z_source
    b1 = truth_params.get("b1", 1.0)

    # Use a clean ell grid for theory (measured ell may have NaN from empty bins)
    ell_clean = np.geomspace(10, 2000, 100)
    cl_kk_theory = compute_theoretical_cl_kappa(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source)
    cl_gg_theory = compute_theoretical_cl_gg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, b1)
    cl_kg_theory = compute_theoretical_cl_kg(cosmo_val, jnp.array(ell_clean), chi_min, chi_max, z_source, b1)

    # 4. Interpolate N_l to ell_clean (reuse ell_in/nell_in from noise plot above)
    nell_at_ell_clean = np.interp(ell_clean, ell_in, nell_in)

    # 5. Plot (Grouped Logic)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- SUBPLOT 1: Kappa Auto ---
    plt.sca(axes[0])
    plot.plot_pow(ell, cl_kk_box, label=r"$C_\ell^{\kappa \kappa}$ (Measured)", color='blue', log=True, ylabel=r"$C_\ell$")
    plot.plot_pow(ell, cl_kk_obs, label=r"$C_\ell^{\kappa \kappa}$ (Observed)", color='gray', alpha=0.5, log=True, ylabel=r"$C_\ell$")
    plt.plot(ell_clean, np.array(cl_kk_theory), '--', color='blue', alpha=0.8, lw=2, label=r"$C_\ell^{\kappa \kappa}$ (Theory)")
    plt.plot(ell_clean, np.array(cl_kk_theory) + nell_at_ell_clean, '--', color='gray', alpha=0.8, lw=2, label=r"Theory + $N_\ell$")

    plt.xlabel(r"Multipole $\ell$")
    plt.title(r"CMB Convergence Auto-Spectra ($C_\ell^{\kappa \kappa}$)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # --- SUBPLOT 2: Galaxy & Cross ---
    plt.sca(axes[1])
    plot.plot_pow(ell, cl_gg, label=r"$C_\ell^{gg}$ (Measured)", color='green', log=True, ylabel=r"$C_\ell$")
    plot.plot_pow(ell, np.abs(cl_kg), label=r"|$C_\ell^{\kappa g}$| (Measured)", color='red', log=True, ylabel=r"$C_\ell$")
    plt.plot(ell_clean, np.array(cl_gg_theory), '--', color='green', alpha=0.8, lw=2, label=r"$C_\ell^{gg}$ (Theory)")
    plt.plot(ell_clean, np.abs(np.array(cl_kg_theory)), '--', color='red', alpha=0.8, lw=2, label=r"|$C_\ell^{\kappa g}$| (Theory)")

    plt.xlabel(r"Multipole $\ell$")
    plt.title(r"Galaxy Auto & Cross Spectra")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(fig_dir / "cl_diagnostics.png", dpi=150)
    plt.close()
    print(f"✓ Diagnostic Spectra: {fig_dir / 'cl_diagnostics.png'}")

# MCMC config
print("\n" + "=" * 80)
print("MCMC CONFIGURATION")
print("=" * 80)

num_warmup = cfg["mcmc"]["num_warmup"]
num_samples = cfg["mcmc"]["num_samples"]  # Samples PER BATCH
num_batches = cfg["mcmc"].get("num_batches", 1)  # Number of batches
num_chains_req = cfg["mcmc"]["num_chains"]

# Auto-adjust num_chains if fewer devices are available
if len(devices) < num_chains_req:
    print(f"\n⚠️  WARNING: Requested {num_chains_req} chains but only {len(devices)} devices available.")
    print(f"   Adjusting num_chains to {len(devices)} to avoid pmap crash.")
    num_chains = max(1, len(devices))
else:
    num_chains = num_chains_req

mclmc_cfg = cfg["mcmc"].get("mclmc", {})
desired_energy_var = float(mclmc_cfg.get("desired_energy_var", 5e-4))
diagonal_precond = bool(mclmc_cfg.get("diagonal_preconditioning", True))
thinning = int(mclmc_cfg.get("thinning", 1))

print("Sampler: MCLMC (Multi-chain + Mini-batch)")
print(f"Chains: {num_chains}")
print(f"Warmup: {num_warmup}")
print(f"Samples per batch: {num_samples}")
print(f"Number of batches: {num_batches}")
print(f"Total samples per chain: {num_samples * num_batches}")
print(f"Total samples (all chains): {num_samples * num_batches * num_chains}")
print(f"Energy var: {desired_energy_var}, Diag precond: {diagonal_precond}")
print(f"Thinning: {thinning}")

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

init_params_ = pmap(init_fn, in_axes=(0, None))(rngs, delta_obs)
init_mesh_ = {k: init_params_[k] for k in ["init_mesh_"]}

print(f"Warming mesh ({cfg['mcmc'].get('mesh_warmup_steps', 2**13)} steps, cosmo/bias fixed)...")
warmup_mesh_fn = pmap(jit(get_mclmc_warmup(
    model.logpdf,
    n_steps=cfg["mcmc"].get("mesh_warmup_steps", 2**13),
    config=None,
    desired_energy_var=1e-6,
    diagonal_preconditioning=True,
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
print(f"  Initial scalar params: fiducial (Omega_m={model.loc_fid['Omega_m']:.4f}, sigma8={model.loc_fid['sigma8']:.4f}, b1={model.loc_fid['b1']:.2f}, ...)")
warmup_all_fn = pmap(jit(get_mclmc_warmup(
    model.logpdf,
    n_steps=num_warmup,
    config=None,
    desired_energy_var=desired_energy_var,
    diagonal_preconditioning=diagonal_precond,
)))

state, config = warmup_all_fn(jr.split(jr.key(43), num_chains), init_params_)

print("✓ Full warmup done")

# Save raw per-chain values
raw_L = jnp.array(config.L)
raw_step_size = jnp.array(config.step_size)

median_L_adapted = float(jnp.median(raw_L))
median_ss = float(jnp.median(raw_step_size))
median_imm = jnp.median(config.inverse_mass_matrix, axis=0)
print(f"  Logdens (median): {float(jnp.median(state.logdensity)):.2f}")
print(f"  Adapted L (median): {median_L_adapted:.6f}")
print(f"  step_size (median): {median_ss:.6f}")

if median_ss < 1.0:
    print("\n⚠️  WARNING: Step size is very small! This indicates poor conditioning or mixing issues.")
    print("   Check if diagonal_preconditioning is enabled or if priors are too tight.")


# Recalculate L (benchmark approach)
eval_per_ess = 1e3
recalc_L = float(0.4 * eval_per_ess / 2 * median_ss)

config = MCLMCAdaptationState(L=recalc_L, step_size=float(median_ss), inverse_mass_matrix=median_imm)
config = tree.map(lambda x: jnp.broadcast_to(x, (num_chains, *jnp.shape(x))), config)

print(f"\n  Recalculated L: {recalc_L:.6f} (was: {median_L_adapted:.6f})")

jnp.savez(config_dir / "warmup_state.npz", **state.position)
utils.ydump(
    {"L": recalc_L, "step_size": median_ss, "eval_per_ess": eval_per_ess},
    config_dir / "warmup_config.yaml",
)

# ========================
# WARMUP VALIDATION TESTS
# ========================
print("\n" + "=" * 80)
print("WARMUP VALIDATION")
print("=" * 80)

# Test 1: Step size consistency across chains
ss_std = float(jnp.std(raw_step_size))
ss_mean = float(jnp.mean(raw_step_size))
ss_rel_std = ss_std / ss_mean if ss_mean > 0 else float('inf')
print("\n1. Step Size:")
print(f"   Values: {raw_step_size}")
print(f"   Mean: {ss_mean:.6f}, Std: {ss_std:.6f}, Rel Std: {ss_rel_std:.4f}")
if ss_rel_std < 0.1:
    print("   ✓ PASS: Step sizes consistent across chains (< 10% relative std)")
else:
    print("   ⚠️  WARN: Step sizes vary significantly across chains")

# Test 2: Logdensity spread (info only - expected to vary after warmup)
logdens = jnp.array(state.logdensity)
ld_std = float(jnp.std(logdens))
print("\n2. Logdensity Spread:")
print(f"   Values: {logdens}")
print(f"   Std: {ld_std:.2f}")

# Test 3: Post-warmup scalar params (physical values) vs truth
print("\n3. Scalar Parameters (post-warmup, physical values vs truth):")
truth_params = cfg.get("truth_params", {})
for param in ["Omega_m_", "sigma8_", "b1_", "b2_", "bs2_", "bn2_"]:
    if param in state.position:
        base_name = param.rstrip("_")
        latent = default_config["latents"].get(base_name, {})
        loc_fid = latent.get("loc_fid", latent.get("loc", 0.0))
        scale_fid = latent.get("scale_fid", 1.0)
        sample_vals = jnp.array(state.position[param])
        physical_vals = loc_fid + sample_vals * scale_fid
        truth = truth_params.get(base_name, "?")
        print(f"   {base_name}: {physical_vals} (truth: {truth})")


# STEP 3: Multi-Chain Mini-Batch Sampling
print("\n" + "=" * 80)
print("STEP 3: MULTI-CHAIN MINI-BATCH SAMPLING")
print("=" * 80)

# Setup sampling function (pmap for parallel chains)
run_fn = pmap(jit(get_mclmc_run(model.logpdf, n_samples=num_samples, thinning=thinning, progress_bar=False)))

print(f"\nRunning {num_chains} chains in parallel, each with {num_batches} sequential batches")
print(f"   Samples per batch: {num_samples}")
print(f"   Total per chain: {num_samples * num_batches}")
print(f"   Total all chains: {num_samples * num_batches * num_chains}")
print(f"   Total evaluations per chain: {num_samples * num_batches * thinning}\n")



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

    # Show MSE per dim for EACH chain
    mse_per_chain = jnp.mean(samples_dict['mse_per_dim'], axis=1)  # shape: (num_chains,)
    print(f"  MSE/dim per chain: {mse_per_chain}")
    print(f"  MSE/dim (median):  {float(jnp.median(mse_per_chain)):.6e}")
    print(f"  Logdensity (median): {float(jnp.median(state.logdensity)):.2f}")
    print(f"  Saved: samples_batch_{batch_idx}.npz")

    # Energy variance validation on first batch
    if batch_idx == 0:
        for i, mse_val in enumerate(mse_per_chain):
            ratio = float(mse_val) / desired_energy_var
            status = "✓" if ratio < 2.0 else "⚠️"
            print(f"\n  📊 Chain {i} Energy Variance: {float(mse_val):.2e} (ratio: {ratio:.2f}) {status}")

        median_mse = float(jnp.median(mse_per_chain))
        median_ratio = median_mse / desired_energy_var
        print(f"\n  📊 Overall (median): {median_mse:.2e}, Desired: {desired_energy_var:.2e}, Ratio: {median_ratio:.2f}")
        if median_ratio < 2.0:
            print("     ✓ PASS: Energy variance matches desired")
        else:
            print(f"     ⚠️  WARN: Ratio = {median_ratio:.1f}x - consider reducing desired_energy_var")

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

# Convergence diagnostics and statistics
print("\n" + "=" * 80)
print("DIAGNOSTICS & STATISTICS")
print("=" * 80)

if hasattr(physical_chains, "print_summary"):
     physical_chains.print_summary()
else:
     # Fallback if chains.py not updated or method missing
     print("chains.print_summary() not found.")

# Viz: Traces
print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

# Check which params are available
available_params = [p for p in comp_params if p in physical_samples]

if available_params:
    plt.figure(figsize=(10, 2 * len(available_params)))
    physical_chains.plot(names=available_params, grid=True)
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
    from getdist import plots

    print("\nGenerating GetDist triangle plot...")

    if available_params:
        # Create MCSamples object using to_getdist
        samples_gd = physical_chains.to_getdist()

        # Prepare markers (truth values as dashed lines)
        markers = {}
        for p in available_params:
            if p in truth_params:
                markers[p] = truth_params[p]

        # Plot
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples_gd], filled=True, title_limit=1, markers=markers)

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
