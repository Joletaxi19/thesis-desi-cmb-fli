from __future__ import annotations  # for Union typing with | in python<3.10

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from pprint import pformat

import jax_cosmo as jc
import numpy as np
import numpyro.distributions as dist
from IPython.display import display
from jax import grad, jacfwd, tree
from jax import numpy as jnp
from jax import random as jr
from jax_cosmo import Cosmology
from jaxpm.painting import cic_paint
from numpyro import deterministic, render_model, sample
from numpyro.handlers import block, condition, seed, trace
from numpyro.infer.util import log_density

from desi_cmb_fli.bricks import (
    get_cosmology,
    kaiser_boost,
    kaiser_model,
    kaiser_posterior,
    lagrangian_weights,
    lin_power_mesh,
    rsd,
    samp2base,
    samp2base_mesh,
)
from desi_cmb_fli.chains import Chains
from desi_cmb_fli.metrics import deconv_paint, powtranscoh, spectrum
from desi_cmb_fli.nbody import lpt, nbody_bf
from desi_cmb_fli.utils import (
    DetruncTruncNorm,
    DetruncUnif,
    cgh2rg,
    ch2rshape,
    nvmap,
    rg2cgh,
    ysafe_dump,
    ysafe_load,
)

default_config = {
    # Mesh and box parameters
    "mesh_shape": 3 * (64,),  # int
    "box_shape": 3 * (320.0,),  # in Mpc/h
    # Evolution
    "a_obs": 0.5,
    "evolution": "lpt",  # kaiser, lpt, nbody
    "nbody_steps": 5,
    "nbody_snapshots": None,
    "lpt_order": 2,
    # Observables
    "gxy_density": 1e-3,  # in galaxy / (Mpc/h)^3
    "observable": "field",  # 'field', TODO: 'powspec' (with poles), 'bispec'
    "los": (0.0, 0.0, 1.0),
    "poles": (0, 2, 4),
    # CMB lensing parameters
    "cmb_enabled": False,  # Enable CMB lensing in forward model and likelihood
    "cmb_lensing_obs": None,  # Observed convergence map (set during conditioning)
    "cmb_noise_nell": None,  # Path to N_ell file or dictionary {'ell': ..., 'N_ell': ...}
    "cmb_field_size_deg": None,  # Angular field size in degrees (None = auto-calc from box size)
    "cmb_field_npix": None,  # Number of pixels for convergence map (None = match mesh_shape)

    "cmb_z_source": 1100.0,  # CMB last scattering surface
    "high_z_mode": "fixed",  # 'fixed', 'taylor', 'exact'

    # Latents
    "precond": "kaiser_dyn",  # direct, fourier, kaiser, kaiser_dyn
    "latents": {
        "Omega_m": {
            "group": "cosmo",
            "label": "{\\Omega}_m",
            "loc": 0.3,
            "scale": 0.15,
            "scale_fid": 0.05,
            "loc_fid": 0.26,
            "low": 0.10,  # Physical minimum (must be > Ω_b ≈ 0.049)
            "high": 0.60,
        },
        "sigma8": {
            "group": "cosmo",
            "label": "{\\sigma}_8",
            "loc": 0.8,
            "scale": 0.15,
            "scale_fid": 0.04,
            "loc_fid": 0.74,
            "low": 0.50,
            "high": 1.10,
        },
        "b1": {
            "group": "bias",
            "label": "{b}_1",
            "loc": 1.2,
            "scale": 0.7,
            "scale_fid": 0.3,
            "loc_fid": 0.9,
            "low": 0.5,
            "high": 4.0,
        },
        "b2": {
            "group": "bias",
            "label": "{b}_2",
            "loc": 0.0,
            "scale": 2.0,
            "scale_fid": 0.7,
            "loc_fid": -0.5,
            "low": -5.0,
            "high": 5.0,
        },
        "bs2": {
            "group": "bias",
            "label": "{b}_{s^2}",
            "loc": 0.0,
            "scale": 2.0,
            "scale_fid": 0.7,
            "loc_fid": 0.6,
            "low": -5.0,
            "high": 5.0,
        },
        "bn2": {
            "group": "bias",
            "label": "{b}_{\\nabla^2}",
            "loc": 0.0,
            "scale": 2.0,
            "scale_fid": 0.7,
            "loc_fid": -0.5,
            "low": -5.0,
            "high": 5.0,
        },
        "init_mesh": {
            "group": "init",
            "label": "{\\delta}_L",
        },
    },
}


def get_model_from_config(config_or_path):
    """
    Factory function to create a FieldLevelModel from a config dictionary or path.
    Handles loading, mesh calculation, and parameter setup (CMB, etc.).

    Args:
        config_or_path (dict or str or Path): Config dict or path to config.yaml.

    Returns:
        tuple: (field_level_model_instance, config_dict)
    """
    if isinstance(config_or_path, str | Path):
        from desi_cmb_fli import utils
        cfg = utils.yload(config_or_path)
    else:
        cfg = config_or_path

    # Build model config
    model_config = default_config.copy()
    model_config["box_shape"] = tuple(cfg["model"]["box_shape"])

    # Cell size & Mesh shape
    cell_size = float(cfg["model"]["cell_size"])
    mesh_shape = [int(round(L / cell_size)) for L in model_config["box_shape"]]
    mesh_shape = [n + 1 if n % 2 != 0 else n for n in mesh_shape] # Ensure even
    model_config["mesh_shape"] = tuple(mesh_shape)

    # Evolution params
    model_config["evolution"] = cfg["model"]["evolution"]
    model_config["lpt_order"] = cfg["model"]["lpt_order"]
    model_config["gxy_density"] = cfg["model"]["gxy_density"]

    # CMB lensing config
    cmb_cfg = cfg.get("cmb_lensing", {})
    if cmb_cfg.get("enabled", False):
        model_config["cmb_enabled"] = True
        # If enabled, setup specific CMB params
        model_config["cmb_field_size_deg"] = None  # Auto-calculate
        model_config["cmb_field_npix"] = int(cmb_cfg["field_npix"]) if "field_npix" in cmb_cfg else None
        model_config["full_los_correction"] = cmb_cfg.get("full_los_correction", False)
        model_config["cmb_z_source"] = float(cmb_cfg.get("z_source", 1100.0))
        model_config["cmb_lensing_obs"] = None # Will be set via conditioning if needed

        # High-Z mode selection
        if "high_z_mode" in cmb_cfg:
            model_config["high_z_mode"] = cmb_cfg["high_z_mode"]
        elif "cache_high_z_cl" in cmb_cfg:
             # Legacy support
             model_config["high_z_mode"] = "fixed" if cmb_cfg["cache_high_z_cl"] else "exact"
        else:
            # Default
            model_config["high_z_mode"] = "fixed"

        if "cmb_noise_nell" in cmb_cfg:
            model_config["cmb_noise_nell"] = cmb_cfg["cmb_noise_nell"]

    return FieldLevelModel(**model_config), model_config


class Model:
    ###############
    # Model calls #
    ###############
    def _model(self, *args, **kwargs):
        raise NotImplementedError

    def model(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def reset(self):
        self.model = self._model

    def __call__(self):
        return self.model()

    def reparam(self, params, inv=False):
        return params

    def _block_det(self, model, hide_base=True, hide_det=True):
        base_name = self.latents.keys()
        if hide_base:
            if hide_det:

                def hide_fn(site):
                    return site["type"] == "deterministic"
            else:

                def hide_fn(site):
                    return site["type"] == "deterministic" and site["name"] in base_name
        else:
            if hide_det:

                def hide_fn(site):
                    return site["type"] == "deterministic" and site["name"] not in base_name
            else:

                def hide_fn(site):
                    return False

        return block(model, hide_fn=hide_fn)

    def predict(
        self,
        rng=42,
        samples=None,
        batch_ndim=0,
        hide_base=True,
        hide_det=True,
        hide_samp=True,
        frombase=False,
    ):
        """
        Run model conditioned on samples.
        * If samples is None, return a single prediction.
        * If samples is an int or tuple, return a prediction of such shape.
        * If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(rng, int):
            rng = jr.key(rng)

        def single_prediction(rng, sample=None):
            if sample is None:
                sample = {}
            # Optionally reparametrize base to sample params
            if frombase:
                sample = self.reparam(sample, inv=True)
                # NOTE: deterministic sites have no effects with handlers.condition, but do with handlers.subsitute

            # Condition then block
            model = condition(self.model, data=sample)
            if hide_samp:
                model = block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = trace(seed(model, rng_seed=rng)).get_trace()
            return {k: v["value"] for k, v in tr.items()}

        if samples is None:
            return single_prediction(rng)

        elif isinstance(samples, int | tuple):
            if isinstance(samples, int):
                samples = (samples,)
            rng = jr.split(rng, samples)
            return nvmap(single_prediction, len(samples))(rng)

        elif isinstance(samples, dict):
            # All item shapes should match on the first batch_ndim dimensions,
            # so take the first item shape
            shape = jnp.shape(next(iter(samples.values())))[:batch_ndim]
            rng = jr.split(rng, shape)
            return nvmap(single_prediction, len(shape))(rng, samples)

    ############
    # Wrappers #
    ############
    def logpdf(self, params):
        """
        A log-density function of the model. In particular, it is the log-*probability*-density function
        with respect to the full set of variables, i.e. E[e^logpdf] = 1.

        For unnormalized log-densities in numpyro, see https://forum.pyro.ai/t/unnormalized-densities/3251/9
        """
        return log_density(self.model, (), {}, params)[0]

    def potential(self, params):
        return -self.logpdf(params)

    def force(self, params):
        return grad(self.logpdf)(params)  # force = - grad potential = grad logpdf

    def trace(self, rng):
        return trace(seed(self.model, rng_seed=rng)).get_trace()

    def seed(self, rng):
        self.model = seed(self.model, rng_seed=rng)

    def condition(self, data=None, frombase=False):
        if data is None:
            data = {}
        # Optionally reparametrize base to sample params
        if frombase:
            data = self.reparam(data, inv=True)
        self.model = condition(self.model, data=data)

    def block(
        self, hide_fn=None, hide=None, expose_types=None, expose=None, hide_base=True, hide_det=True
    ):
        """
        Selectively hides parameters in the model.

        Precedence is given according to the order: hide_fn, hide, expose_types, expose, (hide_base, hide_det).
        Only the set of parameters with the precedence is considered.
        The default call thus hides base and other deterministic sites, for sampling purposes.
        """
        if all(x is None for x in (hide_fn, hide, expose_types, expose)):
            self.model = self._block_det(self.model, hide_base=hide_base, hide_det=hide_det)
        else:
            self.model = block(
                self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose
            )

    def render(self, render_dist=False, render_params=False):
        display(
            render_model(self.model, render_distributions=render_dist, render_params=render_params)
        )

    def partial(self, *args, **kwargs):
        self.model = partial(self.model, *args, **kwargs)

    #################
    # Save and load #
    #################
    def save(self, path):  # with yaml because not array-like
        ysafe_dump(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**ysafe_load(path))


@dataclass
class FourierSpaceGaussian(dist.Distribution):
    arg_constraints = {"loc": dist.constraints.real, "var_k": dist.constraints.positive}
    support = dist.constraints.real

    def __init__(self, loc, var_k, validate_args=None):
        self.loc = loc
        self.var_k = var_k
        super().__init__(batch_shape=(), event_shape=loc.shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.event_shape
        eps = jr.normal(key, shape=shape)
        eps_k = jnp.fft.fftn(eps, axes=(-2, -1))

        N = self.event_shape[0]
        scale = jnp.sqrt(self.var_k) / N
        noise_k = eps_k * scale
        noise_k = noise_k.at[..., 0, 0].set(0.0)  # Zero DC mode

        noise = jnp.fft.ifftn(noise_k, axes=(-2, -1)).real
        return self.loc + noise

    def log_prob(self, value):
        diff = value - self.loc
        diff_k = jnp.fft.fftn(diff, axes=(-2, -1))
        return -0.5 * jnp.sum(jnp.abs(diff_k)**2 / self.var_k, axis=(-2, -1))


@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model,
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    mesh_shape : array_like of int
        Shape of the mesh.
    box_shape : array_like
        Shape of the box in Mpc/h. Typically such that cell lengths would be between 1 and 10 Mpc/h.
    evolution : str
        Evolution model: 'kaiser', 'lpt', 'nbody'.
    a_obs : float
        Scale factor of observations.
    nbody_steps : int
        Number of N-body steps.
        Only used for 'nbody' evolution.
    nbody_snapshots : int or list
        Number or list of N-body snapshots to save. If None, only save last.
        Only used for 'nbody' evolution.
    lpt_order : int
        Order of LPT displacement.
        Only used for 'lpt' evolution.
    observable : str
        Observable: 'field', 'powspec'.
    gxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3
    los : array_like
        Line-of-sight direction. If None, no Redshift Space Distorsion is applied.
    poles : array_like of int
        Power spectrum poles to compute.
        Only used for 'powspec' observable.
    cmb_lensing_obs : array_like or None
        Observed CMB convergence map. If None, no CMB constraint is applied.
    cmb_noise_std : float
        CMB convergence noise std.
    cmb_field_size_deg : float
        Angular field size in degrees for CMB.
    cmb_field_npix : int
        Number of pixels for CMB convergence map.
    cmb_z_source : float
        CMB source redshift (last scattering surface).
    precond : str
        Preconditioning method: 'direct', 'fourier', 'kaiser'.
    latents : dict
        Latent variables configuration.
    """

    # Mesh and box parameters
    mesh_shape: np.ndarray = field(default_factory=lambda: np.array(default_config["mesh_shape"]))
    box_shape: np.ndarray = field(default_factory=lambda: np.array(default_config["box_shape"]))
    # Evolution
    evolution: str = field(default=default_config["evolution"])
    a_obs: float = field(default=default_config["a_obs"])
    nbody_steps: int = field(default=default_config["nbody_steps"])
    nbody_snapshots: int | list = field(default=default_config["nbody_snapshots"])
    lpt_order: int = field(default=default_config["lpt_order"])
    # Observable
    observable: str = field(default=default_config["observable"])
    gxy_density: float = field(default=default_config["gxy_density"])
    los: tuple = field(default=default_config["los"])
    poles: tuple = field(default=default_config["poles"])
    # CMB lensing (with defaults for backward compatibility)
    cmb_lensing_obs: np.ndarray | None = field(default=default_config["cmb_lensing_obs"])
    cmb_noise_nell: str | dict | None = field(default=default_config["cmb_noise_nell"])
    cmb_field_size_deg: float | None = field(default=default_config["cmb_field_size_deg"])
    cmb_field_npix: int | None = field(default=default_config["cmb_field_npix"])
    cmb_z_source: float = field(default=default_config["cmb_z_source"])
    cmb_enabled: bool = field(default=False)  # Explicit flag to enable CMB lensing

    full_los_correction: bool = field(default=False)  # Enable high-z kappa correction
    high_z_mode: str = field(default="fixed")  # 'fixed', 'taylor', 'exact'
    # Latents (required, from default_config)
    precond: str = field(default=default_config["precond"])
    latents: dict = field(default_factory=lambda: default_config["latents"])

    def __post_init__(self):
        self.latents = self._validate_latents()
        self.groups = self._groups(base=True)
        self.groups_ = self._groups(base=False)
        self.labels = self._labels()
        self.loc_fid = self._loc_fid()

        self.mesh_shape = np.asarray(self.mesh_shape)
        # NOTE: if x32, cast mesh_shape into float32 to avoid int32 overflow when computing products
        self.box_shape = np.asarray(self.box_shape, dtype=float)

        # Enforce z_min=0 globally (Observer at face)
        # We assume the box starts at z=0 (chi=0) and extends to box_size[2].
        # We need to find the a_obs that corresponds to the CENTER of the box.
        target_chi = self.box_shape[2] / 2.0

        # Invert chi(a) to find a_obs using fiducial cosmology
        cosmo_fid = get_cosmology(**self.loc_fid)
        # a_of_chi in jax_cosmo
        self.a_obs = float(jc.background.a_of_chi(cosmo_fid, target_chi)[0])

        # Calculate z_max for logging (chi_max = box_size[2])
        chi_max_box = self.box_shape[2]
        a_max = float(jc.background.a_of_chi(cosmo_fid, chi_max_box)[0])
        z_max_box = 1/a_max - 1

        print(f"  Box size: {self.box_shape} Mpc/h")
        print(f"  Target center chi: {target_chi:.2f} Mpc/h")
        print(f"  Computed a_obs: {self.a_obs:.4f} (z_center = {1/self.a_obs - 1:.4f})")
        print(f"  Box extends to chi={chi_max_box:.2f} Mpc/h (z_max = {z_max_box:.4f})")

        self.cell_shape = self.box_shape / self.mesh_shape
        if self.los is not None:
            self.los = np.asarray(self.los)
            self.los = self.los / np.linalg.norm(self.los)

        self.k_funda = 2 * np.pi / np.min(self.box_shape)
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        # 2*pi factors because of Fourier transform definition
        self.gxy_count = self.gxy_density * self.cell_shape.prod()

        if self.cmb_enabled:
            if self.cmb_field_size_deg is None:
                # Box angular size at back: theta = 2 * arctan(L / 2*chi)
                half_angle_rad = np.arctan(self.box_shape[0] / (2.0 * chi_max_box))
                self.cmb_field_size_deg = float(2.0 * half_angle_rad * (180.0 / np.pi))
                print(f"CMB Field Size (Auto): {self.cmb_field_size_deg:.2f} deg")

            if self.cmb_field_npix is None:
                self.cmb_field_npix = int(self.mesh_shape[0])
                print(f"CMB Field Pixels (Auto): {self.cmb_field_npix}")

            # Calculate pixel scale in arcmin
            pixel_scale_deg = self.cmb_field_size_deg / self.cmb_field_npix


            # Load and interpolate N_ell
            if self.cmb_noise_nell is None:
                raise ValueError("cmb_noise_nell is required when cmb_enabled=True")

            if isinstance(self.cmb_noise_nell, str):
                data = np.loadtxt(self.cmb_noise_nell)
                if data.ndim == 1:
                    nell_in = data
                    ell_in = np.arange(len(data))
                else:
                    ell_in, nell_in = data[:, 0], data[:, 1]
            elif isinstance(self.cmb_noise_nell, dict):
                ell_in, nell_in = self.cmb_noise_nell["ell"], self.cmb_noise_nell["N_ell"]
            else:
                ell_in, nell_in = self.cmb_noise_nell[0], self.cmb_noise_nell[1]

            # 2D ell grid
            freq_x = np.fft.fftfreq(self.cmb_field_npix, d=pixel_scale_deg) * 360.0
            freq_y = np.fft.fftfreq(self.cmb_field_npix, d=pixel_scale_deg) * 360.0
            lx, ly = np.meshgrid(freq_y, freq_x, indexing='ij')
            self.ell_grid = np.sqrt(lx**2 + ly**2)  # Store ell_grid for high-z calculation

            # Log-log interpolation of N_ell
            ell_in = np.asarray(ell_in)
            nell_in = np.asarray(nell_in)
            valid_mask = (ell_in > 0) & (np.isfinite(nell_in)) & (nell_in > 0)
            ell_in = ell_in[valid_mask]
            nell_in = nell_in[valid_mask]

            ell_grid_safe = np.maximum(self.ell_grid, 1e-5)
            self.cmb_noise_power_k = np.exp(np.interp(np.log(ell_grid_safe), np.log(ell_in), np.log(nell_in)))

            self.cmb_noise_power_k[0, 0] = np.inf  # Ignore DC mode

            print("CMB Noise Model (N_ell):")
            print(f"  Input N_ell shape: {ell_in.shape}")
            print(f"  Interpolated grid shape: {self.cmb_noise_power_k.shape}")
            print(f"  Min/Max noise power: {self.cmb_noise_power_k.min():.2e} / {self.cmb_noise_power_k.max():.2e}")

            # Cache or Precompute High-Z Correction
            self.cl_high_z_cached = None
            self.high_z_gradients = None

            if self.full_los_correction:
                from desi_cmb_fli.cmb_lensing import compute_theoretical_cl_kappa

                # 1. Compute Fiducial C_l (needed for 'fixed' and 'taylor')
                if self.high_z_mode in ["fixed", "taylor"]:
                    chi_source_fid = float(jc.background.radial_comoving_distance(cosmo_fid, 1.0 / (1.0 + self.cmb_z_source))[0])
                    self.cl_high_z_cached = compute_theoretical_cl_kappa(
                        cosmo_fid,
                        self.ell_grid,
                        chi_max_box,  # chi_min for high-z = chi_max of box
                        chi_source_fid,
                        self.cmb_z_source
                    )
                    print(f"  Cached C_l^{{high-z}} at fiducial (Mode: {self.high_z_mode}). Shape: {self.cl_high_z_cached.shape}")

                # 2. Compute Gradients for Taylor Expansion
                if self.high_z_mode == "taylor":
                    print("  Computing gradients for High-Z Taylor correction...")
                    # Local wrapper to be differentiated
                    def get_theoretical_cl_wrapper(theta):
                        # Theta = [Omega_m, sigma8] (params that vary)
                        Om, s8 = theta
                        # Reconstruct cosmo
                        c_ = get_cosmology(Omega_m=Om, sigma8=s8)
                        chi_source_ = jc.background.radial_comoving_distance(c_, 1.0 / (1.0 + self.cmb_z_source))[0]
                        return compute_theoretical_cl_kappa(
                            c_,
                            self.ell_grid,
                            chi_max_box,
                            chi_source_,
                            self.cmb_z_source
                        )

                    # Fiducial point
                    theta_fid = jnp.array([self.loc_fid["Omega_m"], self.loc_fid["sigma8"]])

                    # Jacobian
                    jac_fn = jacfwd(get_theoretical_cl_wrapper)
                    grads = jac_fn(theta_fid) # Shape: (ell_grid_shape, 2)

                    self.high_z_gradients = {
                        'dCl_dOm': grads[..., 0],
                        'dCl_ds8': grads[..., 1]
                    }

                    print(f"  ✓ Gradients computed. dCl/dOm shape: {self.high_z_gradients['dCl_dOm'].shape}")



    def __str__(self):
        out = ""
        out += "# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"cell_shape:     {list(self.cell_shape)} Mpc/h\n"
        out += f"k_funda:        {self.k_funda:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n"

        if self.full_los_correction:
            out += "full_los_correction: True\n"
        return out




    def _model(self, temp_prior=1.0, temp_lik=1.0):
        cosmo, bias, init = self.prior(temp=temp_prior)
        fields = self.evolve((cosmo, bias, init))
        return self.likelihood(
            cosmology=cosmo,
            bias=bias,
            temp=temp_lik,
            **fields,
        )

    def prior(self, temp=1.0):
        """
        A prior for cosmological model.

        Return base parameters, as reparametrization of sample parameters.
        """
        # Sample, reparametrize, and register cosmology and biases
        tup = ()
        for g in ["cosmo", "bias"]:
            dic = self._sample(self.groups[g])  # sample
            dic = samp2base(dic, self.latents, inv=False, temp=temp)  # reparametrize
            tup += ({k: deterministic(k, v) for k, v in dic.items()},)  # register base params
        cosmo, bias = tup
        cosmology = get_cosmology(**cosmo)

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups["init"][0] + "_"

        bE = 1 + bias["b1"]
        scale, transfer = self._precond_scale_and_transfer(cosmology, bE)
        init[name_] = sample(name_, dist.Normal(0.0, scale))  # sample
        init = samp2base_mesh(
            init, self.precond, transfer=transfer, inv=False, temp=temp
        )  # reparametrize
        init = {k: deterministic(k, v) for k, v in init.items()}  # register base params

        return cosmology, bias, init

    def evolve(self, params: tuple):
        cosmology, bias, init = params

        if self.evolution == "kaiser":
            # Kaiser model
            init_mesh = next(iter(init.values()))
            gxy_mesh = kaiser_model(cosmology, self.a_obs, bE=1 + bias["b1"], init_mesh=init_mesh, los=self.los)
            gxy_mesh = deterministic("gxy_mesh", gxy_mesh)

            # Matter mesh (linear growth, no bias, no RSD)
            matter_mesh = kaiser_model(cosmology, self.a_obs, bE=1.0, init_mesh=init_mesh, los=None)
            matter_mesh = deterministic("matter_mesh", matter_mesh)

            return {"gxy_mesh": gxy_mesh, "matter_mesh": matter_mesh}

        # Create regular grid of particles
        pos = jnp.indices(self.mesh_shape, dtype=float).reshape(3, -1).T

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, self.a_obs, pos, self.box_shape, **bias, **init)
        # TODO: gaussian lagrangian weights

        if self.evolution == "lpt":
            # LPT displacement at a_lpt
            # NOTE: lpt assumes given mesh follows linear spectral power at a=1, and then correct by growth factor for target a_lpt
            cosmology._workspace = {}  # HACK: temporary fix
            dpos, vel = lpt(
                cosmology,
                **init,
                pos=pos,
                a=self.a_obs,
                order=self.lpt_order,
                grad_fd=False,
                lap_fd=False,
            )
            pos += dpos
            pos, vel = deterministic("lpt_pos", pos), vel

        elif self.evolution == "nbody":
            cosmology._workspace = {}  # HACK: temporary fix
            pos, vel = nbody_bf(
                cosmology,
                **init,
                pos=pos,
                a=self.a_obs,
                n_steps=self.nbody_steps,
                grad_fd=False,
                lap_fd=False,
                snapshots=self.nbody_snapshots,
            )
            part = deterministic("nbody_pos", pos), vel
            pos, vel = tree.map(lambda x: x[-1], part)

        # Preserve real-space positions for matter field painting
        pos_real = pos

        # RSD displacement at a_obs
        pos += rsd(cosmology, self.a_obs, vel, self.los)
        pos, vel = deterministic("rsd_pos", pos), vel

        # CIC paint weighted by Lagrangian bias expansion weights
        gxy_mesh = cic_paint(jnp.zeros(self.mesh_shape), pos, lbe_weights)
        gxy_mesh = deconv_paint(gxy_mesh, order=2)
        gxy_mesh = deterministic("gxy_mesh", gxy_mesh)
        # Paint unbiased matter field in real space (before RSD)
        matter_mesh = cic_paint(jnp.zeros(self.mesh_shape), pos_real)
        matter_mesh = deconv_paint(matter_mesh, order=2)
        matter_mesh = matter_mesh / jnp.mean(matter_mesh)
        matter_mesh = deterministic("matter_mesh", matter_mesh)

        return {"gxy_mesh": gxy_mesh, "matter_mesh": matter_mesh}

    def likelihood(self, gxy_mesh, matter_mesh=None, cosmology=None, bias=None, temp=1.0):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        Optionally includes CMB lensing likelihood if cmb_lensing_obs is provided.
        """

        if self.observable == "field":
            # Galaxy likelihood
            obs_mesh = sample("obs", dist.Normal(gxy_mesh, (temp / self.gxy_count) ** 0.5))

            # CMB lensing likelihood (if enabled)
            if self.cmb_enabled and cosmology is not None and bias is not None:
                from desi_cmb_fli.cmb_lensing import (
                    compute_cl_high_z,
                    density_field_to_convergence,
                )

                density_field = matter_mesh

                # Compute predicted convergence
                kappa_pred = density_field_to_convergence(
                    density_field,
                    self.box_shape,
                    cosmology,
                    self.cmb_field_size_deg,
                    self.cmb_field_npix,
                    self.cmb_z_source,
                    box_center_chi=self.box_shape[2] / 2.0,
                )

                # Compute high-z correction using centralized function
                if self.full_los_correction:
                    cl_high_z = compute_cl_high_z(
                        cosmology,
                        self.ell_grid,
                        self.box_shape[2],  # chi_min
                        None,  # chi_max (computed internally)
                        self.cmb_z_source,
                        mode=self.high_z_mode,
                        cl_cached=self.cl_high_z_cached,
                        gradients=self.high_z_gradients,
                        loc_fid=self.loc_fid
                    )
                else:
                    cl_high_z = 0.0

                kappa_pred = deterministic("kappa_pred", kappa_pred)

                # Total variance in Fourier space
                field_area_sr = (self.cmb_field_size_deg * jnp.pi / 180.0)**2
                npix2 = self.cmb_field_npix**2
                total_power_k = self.cmb_noise_power_k + cl_high_z
                variance_k = total_power_k * (npix2**2 / field_area_sr) * temp

                sample("kappa_obs", FourierSpaceGaussian(kappa_pred, variance_k))

            return obs_mesh  # NOTE: mesh is 1+delta_obs

    def reparam(self, params: dict, fourier=True, inv=False, temp=1.0):
        """
        Transform sample params into base params.
        """
        # Extract groups from params
        groups = ["cosmo", "bias", "init"]
        key = tuple([k if inv else k + "_"] for k in groups) + (
            ["*"] + ["~" + k if inv else "~" + k + "_" for k in groups],
        )
        params = Chains(params, self.groups | self.groups_).get(key)  # use chain querying
        cosmo_, bias_, init, rest = (q.data for q in params)
        # cosmo_, bias, init = self._get_by_groups(params, ['cosmo','bias','init'], base=inv)

        # Cosmology and Biases
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)
        bias = samp2base(bias_, self.latents, inv=inv, temp=temp)

        # Initial conditions
        if len(init) > 0:
            cosmology = get_cosmology(**(cosmo_ if inv else cosmo))
            bE = 1 + (bias_["b1"] if inv else bias["b1"])
            _, transfer = self._precond_scale_and_transfer(cosmology, bE)

            if not fourier and inv:
                init = tree.map(lambda x: jnp.fft.rfftn(x), init)

            init = samp2base_mesh(init, self.precond, transfer=transfer, inv=inv, temp=temp)

            if not fourier and not inv:
                init = tree.map(lambda x: jnp.fft.irfftn(x), init)

        return rest | cosmo | bias | init  # possibly update rest

    ###########
    # Getters #
    ###########
    def _validate_latents(self):
        """
        Return a validated latents config.
        """
        new = {}
        for name, conf in self.latents.items():
            new[name] = conf.copy()
            loc, scale = conf.get("loc"), conf.get("scale")
            low, high = conf.get("low"), conf.get("high")
            loc_fid, scale_fid = conf.get("loc_fid"), conf.get("scale_fid")

            assert not (loc is None) ^ (scale is None), (
                f"latent '{name}' not valid: loc and scale must be both provided or both not provided"
            )
            assert not (low is None) ^ (high is None), (
                f"latent '{name}' not valid: low and high must be both provided or both not provided"
            )

            if loc is not None:  # Normal or Truncated Normal prior
                if loc_fid is None:
                    new[name]["loc_fid"] = loc
                if scale_fid is None:
                    new[name]["scale_fid"] = scale

            elif low is not None:  # Uniform prior
                assert low <= high, f"latent '{name}' not valid: low must be lower than high"
                assert low != -jnp.inf and high != jnp.inf, (
                    f"latent '{name}' not valid: low and high must be finite for uniform distribution"
                )
                if loc_fid is None:
                    new[name]["loc_fid"] = (low + high) / 2
                if scale_fid is None:
                    new[name]["scale_fid"] = (high - low) / 12**0.5
        return new

    def _sample(self, names: str | list):
        """
        Sample latent parameters from latents config.
        """
        dic = {}
        names = np.atleast_1d(names)
        for name in names:
            conf = self.latents[name]
            loc, scale = conf.get("loc", None), conf.get("scale", None)
            low, high = conf.get("low", -jnp.inf), conf.get("high", jnp.inf)
            loc_fid, scale_fid = conf["loc_fid"], conf["scale_fid"]

            if loc is not None:
                if low == -jnp.inf and high == jnp.inf:
                    dic[name + "_"] = sample(
                        name + "_", dist.Normal((loc - loc_fid) / scale_fid, scale / scale_fid)
                    )
                else:
                    dic[name + "_"] = sample(
                        name + "_", DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid)
                    )
            else:
                dic[name + "_"] = sample(name + "_", DetruncUnif(low, high, loc_fid, scale_fid))
        return dic

    def _precond_scale_and_transfer(self, cosmo: Cosmology, bE):
        """
        Return scale and transfer fields for linear matter field preconditioning.
        """
        pmeshk = lin_power_mesh(cosmo, self.mesh_shape, self.box_shape)

        if self.precond in ["direct", "fourier"]:
            scale = jnp.ones(self.mesh_shape)
            transfer = pmeshk**0.5

        elif self.precond == "kaiser":
            cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid["b1"]
            boost_fid = kaiser_boost(cosmo_fid, self.a_obs, bE_fid, self.mesh_shape, self.los)
            pmeshk_fid = lin_power_mesh(cosmo_fid, self.mesh_shape, self.box_shape)

            scale = (1 + self.gxy_count * boost_fid**2 * pmeshk_fid) ** 0.5
            transfer = pmeshk**0.5 / scale
            scale = cgh2rg(scale, norm="amp")

        elif self.precond == "kaiser_dyn":
            boost = kaiser_boost(cosmo, self.a_obs, bE, self.mesh_shape, self.los)

            scale = (1 + self.gxy_count * boost**2 * pmeshk) ** 0.5
            transfer = pmeshk**0.5 / scale
            scale = cgh2rg(scale, norm="amp")

        return scale, transfer

    def _groups(self, base=True):
        """
        Return groups from latents config.
        """
        groups = {}
        for name, val in self.latents.items():
            group = val["group"]
            group = group if base else group + "_"
            if group not in groups:
                groups[group] = []
            groups[group].append(name if base else name + "_")
        return groups

    def _labels(self):
        """
        Return labels from latents config
        """
        labs = {}
        for name, val in self.latents.items():
            lab = val["label"]
            labs[name] = lab
            labs[name + "_"] = "\\tilde" + lab
        return labs

    def _loc_fid(self):
        """
        Return fiducial location values from latents config.
        """
        return {k: v["loc_fid"] for k, v in self.latents.items() if "loc_fid" in v}

    ###########
    # Metrics #
    ###########
    def spectrum(self, mesh, mesh2=None, kedges: int | float | list = None, comp=(0, 0), poles=0):
        return spectrum(
            mesh,
            mesh2=mesh2,
            box_shape=self.box_shape,
            kedges=kedges,
            comp=comp,
            poles=poles,
            los=self.los,
        )

    def powtranscoh(self, mesh0, mesh1, kedges: int | float | list = None, comp=(0, 0)):
        return powtranscoh(mesh0, mesh1, box_shape=self.box_shape, kedges=kedges, comp=comp)

    ########################
    # Chains init and load #
    ########################
    def load_runs(self, path: str, start: int, end: int, transforms=None, batch_ndim=2) -> Chains:
        return Chains.load_runs(
            path,
            start,
            end,
            transforms,
            groups=self.groups | self.groups_,
            labels=self.labels,
            batch_ndim=batch_ndim,
        )

    def reparam_chains(self, chains: Chains, fourier=False, batch_ndim=2):
        chains = chains.copy()
        chains.data = nvmap(partial(self.reparam, fourier=fourier), batch_ndim)(chains.data)
        return chains

    def powtranscoh_chains(
        self,
        chains: Chains,
        mesh0,
        name: str = "init_mesh",
        kedges: int | float | list = None,
        comp=(0, 0),
        batch_ndim=2,
    ) -> Chains:
        chains = chains.copy()
        fn = nvmap(lambda x: self.powtranscoh(mesh0, x, kedges=kedges, comp=comp), batch_ndim)
        chains.data["kptc"] = fn(chains.data[name])
        return chains

    def kaiser_post(self, rng, delta_obs, base=False, temp=1.0):
        if jnp.isrealobj(delta_obs):
            delta_obs = jnp.fft.rfftn(delta_obs)

        cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid["b1"]
        means, stds = kaiser_posterior(
            delta_obs, cosmo_fid, bE_fid, self.a_obs, self.box_shape, self.gxy_count, self.los
        )
        post_mesh = rg2cgh(jr.normal(rng, ch2rshape(means.shape)))
        post_mesh = temp**0.5 * stds * post_mesh + means

        init_params = self.loc_fid | {"init_mesh": post_mesh}
        if base:
            return init_params
        else:
            return self.reparam(init_params, inv=True)
