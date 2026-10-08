#!/usr/bin/env python3
"""Early-phase model: the single authoritative physics, prior, constraint and likelihood implementation.

Core tophat jet (forward shock) + reverse shock + host A_V, fit to the pre-flare XRT flux, the early
i band and the XRT spectral index. Everything that scores a parameter vector lives here:

    make_param_defs           parameter set for a prior (final / wide / production), environment and xi layout
    build_model_from_params   VegasAfterglow Model (xi_e -> forward-shock Radiation, xi_e_r -> reverse-shock Radiation)
    chi2_components           chi2 of XRT flux, i-band flux and XRT spectral index
    log_likelihood            -chi2 / 2
    eps_sum_violated          physical constraints eps_e + eps_B <= 1 and eps_e_r + eps_B_r <= 1
    log_prior                 box bounds, Gaussian host A_V, physical constraints
    log_probability           log_prior + log_likelihood (MCMC, optimisers and diagnostics all call this)

Diagnostics (early_phase_diagnostics), optimisers (early_phase_optimize) and the sampler (early_phase_mcmc)
import from here and never re-implement any of it.
"""
import os
from pathlib import Path

import bilby
import numpy as np

from VegasAfterglow import Fitter, ISM, Model, Observer, Radiation, TophatJet, Wind, Scale
from VegasAfterglow.units import Hz, keV, mJy, sec

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import D_L, HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA, REDSHIFT, XRT_FLARE_START_TIME, XRT_NU_HI, XRT_NU_LO
from grb.io import read_data
from grb.params import ENVIRONMENTS, medium_param_defs, make_param_defs as make_final_param_defs
from grb.prior import ParamDefWithPrior
from grb.spectral_index import (
    compute_break_frequencies,
    compute_p_prior_from_spectral_index,
    load_xrt_spectral_index,
)
from grb.utils import flux_error, model_array
from spectral_index_interpolator import get_spectral_index_calculator

I_DATA_ROWS = 40
MODEL_RESOLUTION = (0.1, 0.25, 10)
MODEL_RTOL = 1e-5
DEFAULT_ENVIRONMENT = "wind"
XRT_BAND = (0.3 * keV, 10 * keV)
HOST_AV_PARAM = "A_V"


def load_fit_data(include_spectral_index=False):
    i_data = read_data(
        "i_data",
        correct_galactic_extinction=True,
        add_converted_flux=True,
    ).iloc[:I_DATA_ROWS].copy()

    xrt_data = read_data("xrt").copy()
    xrt_data = xrt_data[xrt_data["time"] < XRT_FLARE_START_TIME]
    xrt_data = xrt_data.reset_index(drop=True)

    xrt_index_data = None
    if include_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
            # Filter to early phase only (before flare)
            mask = xrt_index_data["time"] < XRT_FLARE_START_TIME
            xrt_index_data = {k: v[mask] for k, v in xrt_index_data.items()}
        except Exception as e:
            print(f"Warning: Could not load XRT spectral index data: {e}")
            xrt_index_data = None

    return xrt_data, i_data, xrt_index_data


def make_fitter(xrt_data, i_data, environment=DEFAULT_ENVIRONMENT):
    fitter = Fitter(
        z=REDSHIFT,
        lumi_dist=D_L,
        jet="tophat",
        medium="ism" if environment == "ism" else "wind",
        rvs_shock=True,
        fwd_ssc=False,
        rvs_ssc=False,
        kn=False,
        magnetar=False,
        rtol=MODEL_RTOL,
        resolution=MODEL_RESOLUTION,
        extinction="mw",
    )

    fitter.add_flux(
        band=XRT_BAND,
        t=xrt_data["time"].to_numpy(float) * sec,
        flux=xrt_data["flux"].to_numpy(float),
        err=flux_error(xrt_data),
        num_points=10,
    )

    fitter.add_flux_density(
        nu=float(i_data["frequency_Hz"].iloc[0]) * Hz,
        t=i_data["time"].to_numpy(float) * sec,
        f_nu=i_data["flux_mJy"].to_numpy(float) * mJy,
        err=i_data["flux_mJy_error"].to_numpy(float) * mJy,
    )

    return fitter


PRIORS = ("final", "wide", "production")
XI_MODES = ("both", "reverse", "none")


def xi_flags(mode):
    """(free_xi_e, free_xi_e_r) for an --xi mode: both free, only the reverse shock free, or both fixed at 1."""
    return {"both": (True, True), "reverse": (False, True), "none": (False, False)}[mode]


# Production configuration (2026-10-08) = the historical "final" box with the bound changes established by the
# 2026-10-06/07 diagnostics: tau up to 3000 s (the tau profile peaks at ~400-420 s and falls on both sides),
# reverse-shock eps_e_r / eps_B_r up to 1 (high efficiency preferred; eps_e_r + eps_B_r <= 1 is enforced by
# log_prior), p >= 2.01 and E_iso up to 1e55 erg (1e54 truncated the E_iso * xi_e_r degeneracy).
# xi_e and xi_e_r are 0.01-1, linear, when free (--xi). "final" itself is unchanged.
PRODUCTION_BOUNDS = {"tau_max": 3000.0, "eps_e_r_max": 1.0, "eps_B_r_max": 1.0, "p_min": 2.01, "e_iso_max": 1e55}


def make_param_defs(prior="final", environment=DEFAULT_ENVIRONMENT, tau_max=None, eps_e_r_max=None, eps_B_r_max=None, p_min=None, free_xi_e_r=False, xi_e_r_min=None, e_iso_max=None, free_xi_e=False):
    """Early-phase parameter set.

    final: core jet + reverse shock + host A_V bounds taken from the final fit
        (grb.params.make_param_defs), xi_e / xi_e_r fixed at 1 unless freed.
    wide: the original broad box, with xi_e and xi_e_r free.
    production: final with PRODUCTION_BOUNDS applied (explicit arguments below still take precedence).

    environment (ism / wind / stratified) selects the circumburst density
    parameters (grb.params.medium_param_defs): n_ism, A_star, or A_star + n_ism + n0.
    The wide box keeps its broad n_ism range for ism and uses the grb bounds otherwise.

    tau_max (seconds), eps_e_r_max, eps_B_r_max and e_iso_max, if given, replace only the upper bound of tau,
    eps_e_r, eps_B_r and E_iso respectively, and p_min only the lower bound of p; every other bound and
    prior is untouched. free_xi_e adds the forward-shock xi_e and free_xi_e_r the reverse-shock xi_e_r
    (both 0.01-1, linear, in the wide prior's positions: xi_e after eps_B and before tau, xi_e_r before A_V) as
    free parameters of the final / production prior (the wide prior already has them); otherwise both stay fixed at 1.
    """
    if prior == "production":
        tau_max = PRODUCTION_BOUNDS["tau_max"] if tau_max is None else tau_max
        eps_e_r_max = PRODUCTION_BOUNDS["eps_e_r_max"] if eps_e_r_max is None else eps_e_r_max
        eps_B_r_max = PRODUCTION_BOUNDS["eps_B_r_max"] if eps_B_r_max is None else eps_B_r_max
        p_min = PRODUCTION_BOUNDS["p_min"] if p_min is None else p_min
        e_iso_max = PRODUCTION_BOUNDS["e_iso_max"] if e_iso_max is None else e_iso_max
    param_defs = _make_param_defs(prior, environment)
    if free_xi_e and prior in ("final", "production"):
        tau_pos = [i for i, p in enumerate(param_defs) if p.name == "tau"][0]
        param_defs.insert(tau_pos, ParamDefWithPrior("xi_e", 0.01, 1.0, Scale.LINEAR))
    if free_xi_e_r and prior in ("final", "production"):
        a_v = [i for i, p in enumerate(param_defs) if p.name == "A_V"][0]
        param_defs.insert(a_v, ParamDefWithPrior("xi_e_r", 0.01 if xi_e_r_min is None else xi_e_r_min, 1.0, Scale.LINEAR))
    upper = {"tau": tau_max, "eps_e_r": eps_e_r_max, "eps_B_r": eps_B_r_max, "E_iso": e_iso_max}
    lower = {"p": p_min}
    for i, p in enumerate(param_defs):
        new_upper, new_lower = upper.get(p.name), lower.get(p.name)
        if new_upper is None and new_lower is None:
            continue
        lo = p.lower if new_lower is None else new_lower
        hi = p.upper if new_upper is None else new_upper
        if hi <= lo:
            raise ValueError(f"{p.name} bounds [{lo}, {hi}] are empty")
        param_defs[i] = ParamDefWithPrior(p.name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
    return param_defs


def _make_param_defs(prior, environment):
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}, got {environment!r}")
    if prior in ("final", "production"):
        final_defs = make_final_param_defs(include_flare=False, include_wing=False, environment=environment)
        return [
            ParamDefWithPrior(
                p.name.removesuffix("_core"), p.lower, p.upper, p.scale,
                gaussian_prior=p.gaussian_prior,
            )
            for p in final_defs
        ]
    if prior != "wide":
        raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
    return [
        ParamDefWithPrior("E_iso", 1e50, 1e55, Scale.LOG),
        ParamDefWithPrior("Gamma0", 10, 2000, Scale.LOG),  
        ParamDefWithPrior("theta_c", 1e-3, 0.1, Scale.LOG),
        *([ParamDefWithPrior("n_ism", 1e-4, 1e4, Scale.LOG)] if environment == "ism" else medium_param_defs(environment)),
        ParamDefWithPrior("p", 2.01, 3.5, Scale.LINEAR),  
        ParamDefWithPrior("eps_e", 1e-3, 0.5, Scale.LOG),
        ParamDefWithPrior("eps_B", 1e-6, 1.0, Scale.LOG), 
        ParamDefWithPrior("xi_e", 0.1, 1.0, Scale.LINEAR),
        ParamDefWithPrior("tau", 1, 1e5, Scale.LOG),
        ParamDefWithPrior("p_r", 2.01, 4.0, Scale.LINEAR),
        ParamDefWithPrior("eps_e_r", 1e-3, 1.0, Scale.LOG), 
        ParamDefWithPrior("eps_B_r", 1e-6, 1.0, Scale.LOG), 
        ParamDefWithPrior("xi_e_r", 0.1, 1.0, Scale.LINEAR),
        ParamDefWithPrior(
            HOST_AV_PARAM,
            10 ** (HOST_AV_LOG10_MEAN - 5 * HOST_AV_LOG10_SIGMA),
            10 ** (HOST_AV_LOG10_MEAN + 5 * HOST_AV_LOG10_SIGMA),
            Scale.LOG,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]


def make_priors(param_defs=None, xrt_index_data=None, spectral_index_cooling="slow"):
    """Create bilby priors for parameters with gaussian priors."""
    if param_defs is None:
        param_defs = make_param_defs()
    
    priors = {}
    for param_def in param_defs:
        if not param_def.has_gaussian_prior():
            continue
        
        mean, sigma = param_def.get_prior_mean_sigma()
        param_name = param_def.name
        
        if param_def.scale is Scale.LOG:
            prior_name = f"log10_{param_name}"
            latex_label = f"$\\log_{{10}}({param_name})$"
        else:
            prior_name = param_name
            latex_label = f"${param_name}$"
        
        priors[prior_name] = bilby.core.prior.Gaussian(
            mean,
            sigma,
            name=prior_name,
            latex_label=latex_label,
        )
    
    # Add p prior from spectral index if available
    if xrt_index_data is not None and len(xrt_index_data["beta"]) > 0:
        p_mean, p_sigma = compute_p_prior_from_spectral_index(xrt_index_data, spectral_index_cooling)
        priors["p"] = bilby.core.prior.Gaussian(
            p_mean,
            p_sigma,
            name="p",
            latex_label="$p$",
        )
        print(f"XRT spectral index prior on p: {p_mean:.3f} ± {p_sigma:.3f} (cooling: {spectral_index_cooling})")
    
    return priors


def param_defs_for_saved_result(plot_dir, preferred_prior, preferred_environment, tau_max=None, preferred_xi="both"):
    """Param defs of whichever prior / environment / xi layout matches the labels saved in plot_dir."""
    labels = [line.strip() for line in (Path(plot_dir) / "labels.txt").read_text().splitlines() if line.strip()]
    combos = [(prior, env, xi) for prior in PRIORS for env in ENVIRONMENTS for xi in XI_MODES]
    combos.sort(key=lambda c: c != (preferred_prior, preferred_environment, preferred_xi))
    for prior, env, xi in combos:
        free_xi_e, free_xi_e_r = xi_flags(xi)
        param_defs = make_param_defs(prior, env, tau_max, free_xi_e=free_xi_e, free_xi_e_r=free_xi_e_r)
        if labels == [param_label(p) for p in param_defs]:
            print(f"Saved result matches the '{prior}' prior, '{env}' environment, xi = '{xi}'.")
            return param_defs, env
    raise ValueError(f"Labels in {plot_dir} match no prior/environment/xi combination: {labels}")


def param_label(param_def):
    return f"log10_{param_def.name}" if param_def.scale is Scale.LOG else param_def.name


def physical_param_value(param_def, sampled_value):
    if param_def.scale is Scale.LOG:
        return 10 ** sampled_value
    return sampled_value


def build_model_from_params(params_array, param_defs):
    """Build VegasAfterglow Model directly from parameter array."""
    # Convert params_array to dict
    params = {}
    for i, param_def in enumerate(param_defs):
        value = params_array[i]
        if param_def.scale is Scale.LOG:
            params[param_def.name] = 10 ** value
        else:
            params[param_def.name] = value
    
    # Build model components
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    if "A_star" in params:
        medium = Wind(
            A_star=params["A_star"],
            n_ism=params.get("n_ism"),
            n0=params.get("n0"),
            k_m=params.get("k_m", 2),
        )
    else:
        medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso"],
        Gamma0=params["Gamma0"],
        theta_c=params["theta_c"],
        spreading=True,  # Enable sideways expansion for jet break!
        duration=params["tau"],
    )
    
    fwd_radiation = Radiation(
        eps_e=params["eps_e"],
        eps_B=params["eps_B"],
        p=params["p"],
        xi_e=params.get("xi_e", 1.0),
        ssc=False,
        kn=False,
    )
    
    rvs_radiation = Radiation(
        eps_e=params["eps_e_r"],
        eps_B=params["eps_B_r"],
        p=params["p_r"],
        xi_e=params.get("xi_e_r", 1.0),
        ssc=False,
        kn=False,
    )
    
    return Model(
        jet=jet,
        medium=medium,
        observer=observer,
        fwd_rad=fwd_radiation,
        rvs_rad=rvs_radiation,
        resolutions=MODEL_RESOLUTION,
        rtol=MODEL_RTOL,
    ), params["A_V"]


def evaluate_components(params_array, param_defs, xrt_times, i_times, i_freq):
    """Total/FS/RS model curves from the same Model and host extinction the fit scores.

    Returns (xrt_total, xrt_fwd, xrt_rvs, i_total, i_fwd, i_rvs) in cgs, with the
    host extinction already applied to the i-band curves.
    """
    from grb.extinction import host_extinction_attenuation

    model, A_V = build_model_from_params(params_array, param_defs)
    xrt = model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10)
    i_freqs = np.full_like(i_times, i_freq, dtype=float)
    i_out = model.flux_density(i_times, i_freqs)
    attenuation = host_extinction_attenuation(i_freqs, A_V, REDSHIFT)
    return (
        model_array(xrt.total), model_array(xrt.fwd), model_array(xrt.rvs),
        model_array(i_out.total) * attenuation,
        model_array(i_out.fwd) * attenuation,
        model_array(i_out.rvs) * attenuation,
    )


def _evaluate_fn(params_array, param_defs):
    return lambda xrt_times, i_times, i_freq: evaluate_components(params_array, param_defs, xrt_times, i_times, i_freq)


def model_spectral_index(model, times):
    """Local synchrotron slope beta = dln F_nu / dln nu across the XRT band (F_nu ~ nu^beta)."""
    times = np.asarray(times, dtype=float)
    f_lo = model_array(model.flux_density(times, XRT_NU_LO * np.ones_like(times)).total)
    f_hi = model_array(model.flux_density(times, XRT_NU_HI * np.ones_like(times)).total)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(f_hi / f_lo) / np.log(XRT_NU_HI / XRT_NU_LO)


def chi2_components(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """chi2 split into (XRT flux, i-band flux, XRT spectral index); the last is inf if the model slope is not finite."""
    # Build model from parameters
    model, A_V = build_model_from_params(params_array, param_defs)
    
    # XRT flux predictions (no host extinction for X-rays)
    xrt_times = xrt_data["time"].to_numpy(float)
    xrt_model_flux = model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10).total
    
    # i-band flux density predictions with host extinction
    i_times = i_data["time"].to_numpy(float)
    i_freqs = i_data["frequency_Hz"].to_numpy(float)
    i_model_flux_density = model.flux_density(i_times, i_freqs).total
    
    # Apply host extinction to optical data
    from grb.extinction import host_extinction_attenuation
    attenuation = host_extinction_attenuation(i_freqs, A_V, REDSHIFT)
    i_model_flux_density_attenuated = i_model_flux_density * attenuation
    
    # XRT chi2
    xrt_diff = xrt_data["flux"].to_numpy(float) - xrt_model_flux
    chi2_xrt = np.sum((xrt_diff / flux_error(xrt_data)) ** 2)
    
    # i-band chi2
    i_diff = i_data["flux_mJy"].to_numpy(float) - i_model_flux_density_attenuated / mJy
    chi2_i = np.sum((i_diff / i_data["flux_mJy_error"].to_numpy(float)) ** 2)
    
    # Add spectral index constraint using smooth transitions
    chi2_si = 0.0
    if xrt_index_data is not None:
        # Extract physical parameters for break frequency calculations
        params_dict = {}
        for idx, param_def in enumerate(param_defs):
            value = params_array[idx]
            if param_def.scale is Scale.LOG:
                params_dict[param_def.name] = 10 ** value
            else:
                params_dict[param_def.name] = value
        
        # The analytic Granot & Sari break frequencies below are ISM-only; for wind /
        # stratified media use the model's own local slope across the XRT band.
        if "A_star" in params_dict:
            beta_models = model_spectral_index(model, xrt_index_data["time"])
            if not np.all(np.isfinite(beta_models)):
                return float(chi2_xrt), float(chi2_i), np.inf
            for beta_model, beta_obs, beta_err_low, beta_err_high in zip(
                beta_models,
                xrt_index_data["beta"],
                xrt_index_data["beta_err_low"],
                xrt_index_data["beta_err_high"],
            ):
                beta_err = beta_err_high if beta_model > beta_obs else beta_err_low
                chi2_si += ((beta_obs - beta_model) / beta_err) ** 2
            return float(chi2_xrt), float(chi2_i), float(chi2_si)

        # Get spectral index calculator (uses precomputed interpolation)
        if spectral_index_cooling == "smooth":
            calc = get_spectral_index_calculator()
        
        # XRT center frequency (~1.2 keV)
        nu_xrt = 3e17  # Hz
        
        for t_obs, beta_obs, beta_err_low, beta_err_high in zip(
            xrt_index_data["time"],
            xrt_index_data["beta"],
            xrt_index_data["beta_err_low"],
            xrt_index_data["beta_err_high"]
        ):
            if spectral_index_cooling == "smooth":
                # Compute break frequencies at observation time
                breaks = compute_break_frequencies(params_dict, REDSHIFT, t_obs)
                
                # Use smooth spectral index calculation
                beta_model = calc.beta_at_frequency(
                    nu_xrt, breaks["nu_m"], breaks["nu_c"], params_dict["p"]
                )
            else:
                # Use simple step function (backward compatibility)
                p_value = params_dict["p"]
                if spectral_index_cooling == "slow":
                    beta_model = (p_value - 1.0) / 2.0
                elif spectral_index_cooling == "fast":
                    beta_model = p_value / 2.0
                elif spectral_index_cooling == "both":
                    beta_slow = (p_value - 1.0) / 2.0
                    beta_fast = p_value / 2.0
                    beta_model = 0.5 * (beta_slow + beta_fast)
                else:
                    raise ValueError(f"Unknown cooling regime: {spectral_index_cooling}")
            
            # Use appropriate error (asymmetric errors)
            beta_err = beta_err_high if beta_model > beta_obs else beta_err_low
            chi2_si += ((beta_obs - beta_model) / beta_err) ** 2
    
    return float(chi2_xrt), float(chi2_i), float(chi2_si)


def chi2_with_spectral_index(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Compute chi2 including XRT spectral index constraints."""
    return float(sum(chi2_components(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)))


def log_likelihood(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Log-likelihood function for emcee."""
    chi2 = chi2_with_spectral_index(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    return -0.5 * chi2


EPS_SUM_PAIRS = (("eps_e", "eps_B"), ("eps_e_r", "eps_B_r"))
EPS_SUM_TOLERANCE = 1e-9


def eps_sum_violated(params_array, param_defs):
    """True if any shock has eps_e + eps_B > 1 (more than all of the shocked internal energy)."""
    values = {
        param_def.name: 10 ** value if param_def.scale is Scale.LOG else value
        for param_def, value in zip(param_defs, params_array)
    }
    return any(
        electrons in values and magnetic in values and values[electrons] + values[magnetic] > 1.0 + EPS_SUM_TOLERANCE
        for electrons, magnetic in EPS_SUM_PAIRS
    )


def log_prior(params_array, param_defs):
    """Log-prior function for emcee: box bounds, Gaussian host A_V, and eps_e + eps_B <= 1 per shock."""
    log_prob = 0.0
    
    for param_def, value in zip(param_defs, params_array):
        # Check bounds
        sampled_lower = np.log10(param_def.lower) if param_def.scale is Scale.LOG else param_def.lower
        sampled_upper = np.log10(param_def.upper) if param_def.scale is Scale.LOG else param_def.upper
        
        if not (sampled_lower <= value <= sampled_upper):
            return -np.inf
        
        # Apply Gaussian priors if defined
        if isinstance(param_def, ParamDefWithPrior) and param_def.has_gaussian_prior():
            mean, sigma = param_def.get_prior_mean_sigma()
            log_prob += -0.5 * ((value - mean) / sigma) ** 2

    if eps_sum_violated(params_array, param_defs):
        return -np.inf

    return log_prob


def log_probability(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Log-probability function for emcee."""
    lp = log_prior(params_array, param_defs)
    if not np.isfinite(lp):
        return -np.inf
    
    try:
        ll = log_likelihood(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    except (ValueError, FloatingPointError):
        return -np.inf
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# Global context for multiprocessing
_EARLY_PHASE_CONTEXT = None

def _init_early_phase_context(context):
    """Initialize global context for worker processes."""
    global _EARLY_PHASE_CONTEXT
    _EARLY_PHASE_CONTEXT = context

def _log_probability_from_context(params_array):
    """Log-probability function that uses global context."""
    return log_probability(params_array, *_EARLY_PHASE_CONTEXT)


def _sampled_bounds(param_defs):
    bounds = []
    for param_def in param_defs:
        if param_def.scale is Scale.LOG:
            bounds.append((np.log10(param_def.lower), np.log10(param_def.upper)))
        else:
            bounds.append((param_def.lower, param_def.upper))
    return np.array(bounds)


def _neg_log_probability_from_context(params_array):
    lp = _log_probability_from_context(params_array)
    return -lp if np.isfinite(lp) else 1e30


def posterior_decomposition(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Posterior split into its chi2 / prior / likelihood pieces, built from the same calls the fit uses."""
    chi2_xrt, chi2_i, chi2_si = chi2_components(
        params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling
    )
    chi2_total = chi2_xrt + chi2_i + chi2_si
    lp = log_prior(params_array, param_defs)
    ll = log_likelihood(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    return {
        "chi2_XRT_flux": chi2_xrt,
        "chi2_i_band": chi2_i,
        "chi2_XRT_spectral_index": chi2_si,
        "chi2_total": chi2_total,
        "log_prior": lp,
        "log_likelihood": ll,
        "log_posterior": lp + ll,
    }


# ---------------------------------------------------------------------------------------------------------------
# Sampling coordinates: an optional change of the sampler's coordinates only (2026-10-08).
#
# "physical": the sampler moves in the sampled space above (log10 for LOG parameters, linear otherwise).
# "q_E_xir":  log10_E_iso is replaced, in the same slot, by q_E_xir = log10(E_iso * xi_e_r) = log10_E_iso + log10(xi_e_r);
#             xi_e_r stays a coordinate. Before every evaluation the vector is mapped back
#             (log10_E_iso = q_E_xir - log10(xi_e_r)) and handed to the unchanged log_probability, whose log_prior
#             still enforces the E_iso and xi_e_r box bounds on the reconstructed values. The map
#             (log10_E_iso, xi_e_r) -> (q_E_xir, xi_e_r) has Jacobian determinant 1, so the prior measure (uniform in
#             log10_E_iso, linear in xi_e_r) is exactly the same; q_E_xir has no box of its own.
# "u_r":      p_r is replaced, in the same slot, by u_r = log10(p_r - 2) (p_r = 2 + 10**u_r). The physical prior on
#             p_r (uniform between its bounds) is kept exactly by adding log|dp_r/du_r| = ln(ln 10) + u_r ln 10
#             (sampling_log_jacobian) to the log target; the p_r bounds are still enforced by log_prior on the
#             reconstructed p_r, so u_r runs over (log10(p_r_min - 2), log10(p_r_max - 2)] = (-inf, log10 1.5] for
#             the production bounds [2, 3.5] (the 10**u_r Jacobian makes the open lower end integrable).
# ---------------------------------------------------------------------------------------------------------------
SAMPLING_COORDS = ("physical", "q_E_xir", "u_r")
Q_E_XIR = "q_E_xir"
U_R = "u_r"
_LN_LN10 = float(np.log(np.log(10.0)))


def sampling_labels(labels, coords):
    """Labels of the sampler's coordinates for physical-space labels `labels`."""
    if coords == "physical":
        return list(labels)
    if coords == Q_E_XIR:
        return [Q_E_XIR if l == "log10_E_iso" else l for l in labels]
    if coords == U_R:
        if "p_r" not in labels:
            raise ValueError("u_r coordinates need a free p_r")
        return [U_R if l == "p_r" else l for l in labels]
    raise ValueError(f"sampling coordinates must be one of {SAMPLING_COORDS}, got {coords!r}")


def sampling_log_jacobian(y, labels, coords):
    """log |d(physical sampled vector) / d(sampler coordinates)| for vector(s) y (..., ndim): the term added to
    log_probability so that the prior on the physical parameters is unchanged. 0 for "physical" and "q_E_xir"."""
    y = np.asarray(y, dtype=float)
    if coords == U_R:
        return _LN_LN10 + y[..., labels.index("p_r")] * np.log(10.0)
    sampling_labels(labels, coords)
    return np.zeros(y.shape[:-1]) if y.ndim > 1 else 0.0


def _q_indices(labels):
    if "log10_E_iso" not in labels or "xi_e_r" not in labels:
        raise ValueError("q_E_xir coordinates need log10_E_iso and a free xi_e_r")
    return labels.index("log10_E_iso"), labels.index("xi_e_r")


def to_sampling_coords(x, labels, coords):
    """Physical sampled-space vector(s) x (..., ndim) -> sampler coordinates."""
    y = np.array(x, dtype=float)
    if coords == "physical":
        return y
    sampling_labels(labels, coords)
    if coords == U_R:
        pr_i = labels.index("p_r")
        with np.errstate(divide="ignore", invalid="ignore"):
            y[..., pr_i] = np.log10(np.subtract(y[..., pr_i], 2.0))
        return y
    e_i, xi_i = _q_indices(labels)
    y[..., e_i] = y[..., e_i] + np.log10(y[..., xi_i])
    return y


def from_sampling_coords(y, labels, coords):
    """Sampler coordinates (..., ndim) -> physical sampled-space vector(s). Non-positive xi_e_r gives a non-finite
    log10_E_iso, which log_prior rejects together with the xi_e_r bound itself."""
    x = np.array(y, dtype=float)
    if coords == "physical":
        return x
    sampling_labels(labels, coords)
    if coords == U_R:
        pr_i = labels.index("p_r")
        x[..., pr_i] = np.add(2.0, np.power(10.0, x[..., pr_i]))
        return x
    e_i, xi_i = _q_indices(labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        x[..., e_i] = x[..., e_i] - np.log10(x[..., xi_i])
    return x
