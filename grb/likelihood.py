"""Likelihood, prior, and posterior for the combined afterglow fit.

Total flux = core jet (forward + optional reverse shock) + optional wing jet
+ optional Norris flare in the XRT band, spectrally extrapolated to optical.
The likelihood is a chi-squared over XRT plus each optical dataset separately,
plus an optional XRT spectral-index term.
"""
import numpy as np

from VegasAfterglow import Scale

from .const import (
    REDSHIFT,
    SI_FLARE_FRAC_MAX,
    XRT_BAND,
    XRT_NU_HI,
    XRT_NU_LO,
)
from .extinction import host_extinction_attenuation
from .functions import norris_flare
from .modeling import make_core_model, make_wing_model
from .params import dataset_cal_factor
from .prior import ParamDefWithPrior
from .utils import model_array


def spectral_index_model(core_model, wing_model, params, times, include_flare,
                         flare_frac_max=SI_FLARE_FRAC_MAX):
    """Model XRT spectral index and flux-contribution selection at given times.

    Returns (beta_model, keep), where beta_model is the local synchrotron slope
    beta = dln(F_nu)/dln(nu) of the summed core+wing flux between XRT_NU_LO and
    XRT_NU_HI (F_nu ∝ nu^beta, matching load_xrt_spectral_index's beta = 1 - Gamma
    convention). keep is a boolean mask marking points where core+wing dominate the
    XRT flux (flare fraction <= flare_frac_max); the phenomenological flare has no
    clean synchrotron slope, so flare-dominated points are excluded.
    """
    times = np.asarray(times, dtype=float)

    def synch_density(nu):
        f = model_array(core_model.flux_density(times, nu * np.ones_like(times)).total).copy()
        if wing_model is not None:
            f = f + model_array(wing_model.flux_density(times, nu * np.ones_like(times)).total)
        return f

    f_lo = synch_density(XRT_NU_LO)
    f_hi = synch_density(XRT_NU_HI)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_model = np.log(f_hi / f_lo) / np.log(XRT_NU_HI / XRT_NU_LO)

    keep = np.isfinite(beta_model)

    # Flux-contribution selection: drop points where the flare dominates the XRT flux
    if include_flare and "t_start_flare" in params:
        cw_band = model_array(core_model.flux(times, XRT_BAND[0], XRT_BAND[1], 10).total).copy()
        if wing_model is not None:
            cw_band = cw_band + model_array(wing_model.flux(times, XRT_BAND[0], XRT_BAND[1], 10).total)
        flare_band = norris_flare(times, params["t_start_flare"], params["tau_rise_flare"],
                                  params["tau_decay_flare"], params["A_flare"])
        total_band = cw_band + flare_band
        flare_frac = np.divide(flare_band, total_band, out=np.zeros_like(total_band),
                               where=total_band > 0)
        keep &= flare_frac <= flare_frac_max

    return beta_model, keep


def spectral_index_chi2(core_model, wing_model, params, xrt_index_data, include_flare,
                        flare_frac_max=SI_FLARE_FRAC_MAX):
    """Chi2 from the XRT spectral (photon) index (see spectral_index_model)."""
    beta_model, keep = spectral_index_model(
        core_model, wing_model, params, xrt_index_data["time"], include_flare, flare_frac_max
    )
    if not np.any(keep):
        return 0.0

    beta_obs = xrt_index_data["beta"]
    # Asymmetric errors (following early_phase.py)
    err = np.where(beta_model > beta_obs,
                   xrt_index_data["beta_err_high"], xrt_index_data["beta_err_low"])
    resid = (beta_obs - beta_model) / err
    return float(np.sum(resid[keep] ** 2))


def compute_model_flux_all_bands(params, xrt_data, optical_datasets, include_flare, include_wing,
                                 xrt_index_data=None):
    """Compute model flux for XRT and all optical bands.

    Returns (xrt_flux, optical_fluxes, si_chi2), where si_chi2 is the XRT
    spectral-index chi2 (0.0 when xrt_index_data is None).
    """
    # Core model
    core_model = make_core_model(params)

    # Wing model
    if include_wing and "E_iso_wing" in params:
        wing_model = make_wing_model(params)
    else:
        wing_model = None

    # XRT flux (xrt_data is now a dict with 'time', 'flux', 'flux_error')
    xrt_times = xrt_data['time']
    xrt_core = core_model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10)
    xrt_flux = model_array(xrt_core.total).copy()  # Make writable copy

    if wing_model:
        xrt_wing = wing_model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10)
        xrt_flux += model_array(xrt_wing.total)

    # Flare in XRT
    if include_flare and "t_start_flare" in params:
        flare_temporal = norris_flare(xrt_times, params["t_start_flare"],
                                     params["tau_rise_flare"], params["tau_decay_flare"],
                                     params["A_flare"])
        xrt_flux += flare_temporal

    # Optical fluxes (all bands)
    optical_fluxes = []
    for dataset in optical_datasets:
        times = dataset['time']
        nu = dataset['frequency']

        # Core + Wing
        nu_array = nu * np.ones_like(times)
        opt_core = core_model.flux_density(times, nu_array)
        opt_flux = model_array(opt_core.total).copy()  # Make writable copy

        if wing_model:
            opt_wing = wing_model.flux_density(times, nu_array)
            opt_flux += model_array(opt_wing.total)

        # Flare contribution (with spectral scaling)
        if include_flare and "t_start_flare" in params:
            # XRT band for normalization
            nu_xrt_min = XRT_NU_LO
            nu_xrt_max = XRT_NU_HI
            beta_flare = params.get("flare_beta", 0.8)

            flare_temporal = norris_flare(times, params["t_start_flare"],
                                         params["tau_rise_flare"], params["tau_decay_flare"],
                                         params["A_flare"])

            # Convert to flux density at optical frequency
            if abs(beta_flare - 1.0) < 0.01:
                K = flare_temporal / np.log(nu_xrt_max / nu_xrt_min)
            else:
                K = flare_temporal * (1 - beta_flare) / (nu_xrt_max**(1-beta_flare) - nu_xrt_min**(1-beta_flare))

            # K is array (time-dependent), nu^-beta is scalar (freq-dependent)
            opt_flux = opt_flux + K * nu**(-beta_flare)

        # Apply host extinction
        if "A_V" in params:
            attenuation = host_extinction_attenuation(nu, params["A_V"], REDSHIFT)
            opt_flux *= attenuation

        # Convert to mJy and apply cross-calibration scale (7DT reference => 1)
        opt_flux_mJy = opt_flux * 1e26 * dataset_cal_factor(params, dataset['name'])
        optical_fluxes.append(opt_flux_mJy)

    # XRT spectral-index chi2 (reuses the already-built core/wing models)
    si_chi2 = 0.0
    if xrt_index_data is not None and len(xrt_index_data["time"]) > 0:
        si_chi2 = spectral_index_chi2(core_model, wing_model, params, xrt_index_data, include_flare)

    return xrt_flux, optical_fluxes, si_chi2


def log_likelihood(theta, param_defs, xrt_data, optical_datasets, include_flare, include_wing,
                   xrt_index_data=None):
    """Log likelihood for all data"""
    params = {}
    for param_def, value in zip(param_defs, theta):
        if param_def.scale is Scale.LOG:
            params[param_def.name] = 10 ** value
        else:
            params[param_def.name] = value

    try:
        xrt_model, optical_models, si_chi2 = compute_model_flux_all_bands(
            params, xrt_data, optical_datasets, include_flare, include_wing, xrt_index_data
        )

        chi2 = 0.0

        # XRT chi-squared (xrt_data is now a dict with arrays)
        xrt_flux = xrt_data['flux']
        xrt_err = xrt_data['flux_error']
        xrt_residuals = xrt_flux - xrt_model
        chi2 += np.sum((xrt_residuals / xrt_err) ** 2)

        # Optical chi-squared (all bands)
        for dataset, model_flux in zip(optical_datasets, optical_models):
            data_flux = dataset['flux_mJy']
            data_err = dataset['flux_err']
            chi2 += np.sum(((data_flux - model_flux) / data_err) ** 2)

        # XRT spectral-index chi-squared
        chi2 += si_chi2

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    except:
        return -np.inf


def log_prior(theta, param_defs):
    """Log prior"""
    log_prob = 0.0

    for param_def, value in zip(param_defs, theta):
        sampled_lower = np.log10(param_def.lower) if param_def.scale is Scale.LOG else param_def.lower
        sampled_upper = np.log10(param_def.upper) if param_def.scale is Scale.LOG else param_def.upper

        if not (sampled_lower <= value <= sampled_upper):
            return -np.inf

        if isinstance(param_def, ParamDefWithPrior) and param_def.has_gaussian_prior():
            mean, sigma = param_def.get_prior_mean_sigma()
            log_prob += -0.5 * ((value - mean) / sigma) ** 2

    return log_prob


def log_probability(theta, param_defs, xrt_data, optical_datasets, include_flare, include_wing,
                    xrt_index_data=None):
    """Combined log probability"""
    lp = log_prior(theta, param_defs)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, param_defs, xrt_data, optical_datasets, include_flare, include_wing,
                        xrt_index_data)
    return lp + ll
