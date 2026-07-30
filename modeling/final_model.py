#!/usr/bin/env python3
"""
Final Model - Complete Fit with All Data
Combines:
- Core jet + Reverse shock (from early_phase)
- Norris flare (from partial_data with Norris function)
- Wing jet (from late_phase)
- ALL optical data: i-band + Leavitt Rc/Ic + SDT
"""
from pathlib import Path
from datetime import datetime
import argparse
import os
import multiprocessing as mp
from multiprocessing import Pool

# Keep each worker single-threaded so 64 pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib.pyplot as plt
import numpy as np
import emcee

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet, ParamDef, Scale
from VegasAfterglow.units import keV

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import D_L, REDSHIFT, TRIGGER_TIME
from grb.io import read_data, filter_data
from utils import (
    HOST_AV_LOG10_MEAN,
    HOST_AV_LOG10_SIGMA,
    ParamDefWithPrior,
    host_extinction_attenuation,
    load_xrt_spectral_index,
    model_array,
    plot_corner,
    top_k_samples,
    xrt_flux_error,
)
from final_model_plotting import plot_light_curves, plot_spectral_index_comparison

PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"
XRT_BAND = (0.3 * keV, 10.0 * keV)
MODEL_RESOLUTIONS = (0.1, 0.25, 10)

# XRT spectral-index constraint: local synchrotron slope is measured between these
# two frequencies (0.3 and 10 keV in Hz). The constraint is applied only at index
# points where the flare contributes less than SI_FLARE_FRAC_MAX of the XRT flux,
# since the phenomenological flare has no clean synchrotron slope.
XRT_NU_LO = 7.25e16
XRT_NU_HI = 2.42e18
SI_FLARE_FRAC_MAX = 0.5


def seconds_from_trigger(date_obs):
    """Convert date_obs string to seconds from trigger"""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.strptime(str(date_obs), fmt) - TRIGGER_TIME).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date_obs format: {date_obs}")


def norris_flare(t, t_start, tau_rise, tau_decay, amplitude):
    """Norris function for GRB flare profile"""
    flux = np.zeros_like(t, dtype=float)
    mask = t > t_start
    
    if np.any(mask):
        dt = t[mask] - t_start
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            flux[mask] = amplitude * np.exp(-tau_rise / dt - dt / tau_decay)
        flux[~np.isfinite(flux)] = 0.0
    
    return flux


def load_all_optical_data():
    """Load ALL optical and XRT data (following late_phase.py structure)"""
    # XRT data
    xrt_data = read_data("xrt")
    xrt_dict = {
        'time': xrt_data['time'].to_numpy(float),
        'flux': xrt_data['flux'].to_numpy(float),
        'flux_error': xrt_flux_error(xrt_data),
    }
    
    optical_datasets = []
    
    # 1. i-band data (primary)
    i_data = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    optical_datasets.append({
        'name': 'i-band',
        'frequency': float(i_data['frequency_Hz'].iloc[0]),
        'time': i_data['time'].to_numpy(float),
        'flux_mJy': i_data['flux_mJy'].to_numpy(float),
        'flux_err': i_data['flux_mJy_error'].to_numpy(float),
    })
    
    # 2. Leavitt Rc and Ic data
    circular = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
    for filter_name in ("Rc", "Ic"):
        data = filter_data(circular, filter_name=filter_name, 
                          facility_name="Leavitt", remove_upper_limits=True)
        if len(data) > 0:
            optical_datasets.append({
                'name': f'Leavitt_{filter_name}',
                'frequency': float(data['frequency_Hz'].iloc[0]),
                'time': data['time'].to_numpy(float),
                'flux_mJy': data['flux_mJy'].to_numpy(float),
                'flux_err': data['flux_mJy_error'].to_numpy(float),
            })
    
    # 3. SDT/7DT data - Each filter is a separate dataset!
    sdt_data = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
    sdt_data = sdt_data[~sdt_data["is_upper_limit"].astype(bool)].copy()
    for _, row in sdt_data.iterrows():
        optical_datasets.append({
            'name': f'7DT_{row["filter_name"]}',
            'frequency': float(row["frequency_Hz"]),
            'time': np.array([seconds_from_trigger(row["date_obs"])]),
            'flux_mJy': np.array([float(row["flux_mJy"])]),
            'flux_err': np.array([float(row["flux_mJy_error"])]),
        })
    
    return xrt_dict, optical_datasets


def make_core_model(params):
    """Core jet with reverse shock"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_core"],
        Gamma0=params["Gamma0_core"],
        theta_c=params["theta_c_core"],
        spreading=True,
        duration=params.get("tau", 10.0),
    )
    fwd_radiation = Radiation(
        eps_e=params["eps_e"],
        eps_B=params["eps_B"],
        p=params["p"],
        xi_e=params["xi"],
        ssc=False,
        kn=False,
    )
    
    rvs_radiation = None
    if "p_r" in params and "eps_e_r" in params and "eps_B_r" in params:
        rvs_radiation = Radiation(
            eps_e=params["eps_e_r"],
            eps_B=params["eps_B_r"],
            p=params["p_r"],
            xi_e=params.get("xi_r", params["xi"]),
            ssc=False,
            kn=False,
        )
    
    return Model(jet=jet, medium=medium, observer=observer, 
                 fwd_rad=fwd_radiation, rvs_rad=rvs_radiation, 
                 resolutions=MODEL_RESOLUTIONS)


def make_wing_model(params):
    """Wing jet (no reverse shock, with spreading for late-time emission)"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_wing"],
        Gamma0=params["Gamma0_wing"],
        theta_c=params["theta_c_wing"],
        spreading=True,  # Enable spreading to maintain flux at late times
        duration=params.get("tau", 10.0),
    )
    radiation = Radiation(
        eps_e=params.get("eps_e_wing", params["eps_e"]),
        eps_B=params.get("eps_B_wing", params["eps_B"]),
        p=params.get("p_wing", params["p"]),
        xi_e=params.get("xi_wing", params["xi"]),
        ssc=False,
        kn=False,
    )
    return Model(jet=jet, medium=medium, observer=observer, 
                 fwd_rad=radiation, resolutions=MODEL_RESOLUTIONS)


def make_param_defs(include_flare=True, include_wing=True):
    """Parameter definitions

    Bounds retuned (2026-07-30) from the joint re-optimization of the
    final_flare_wing_20260724_171919 best fit inside a widened box
    (logL -577.6 -> -548.0, total chi2 1154.9 -> 1095.8 on 232 points):
    - tau_rise_flare and p_wing: the improved optimum sits OUTSIDE the old
      bounds (25.5 s < 30 s; 3.06 > 2.9), so widening these is required.
    - theta_c_core, n_ism, eps_B, E_iso_wing, eps_e_wing, theta_c_wing:
      posterior modes hug the old walls; widened so the posterior can close.
    - p and the eps_B lower range are deliberately NOT opened further:
      chasing the observed XRT photon index (~1.88, vs model floor ~2.04)
      via low eps_B / high p was tested and loses badly (dlogL <= -620) -
      the spectral-index tension (chi2 ~ 88 for 45 pts) is a model
      limitation, not a bounds artifact.
    """
    params = [
        # Core jet (narrow range to avoid bimodal distribution)
        ParamDefWithPrior("E_iso_core", 5e51, 1e53, Scale.LOG),
        ParamDefWithPrior("Gamma0_core", 300, 1100, Scale.LOG),  # Extended for narrow jets
        ParamDefWithPrior("theta_c_core", 0.001, 0.08, Scale.LOG),  # mode 0.039 hugged old 0.04 wall

        # Environment & forward shock microphysics
        ParamDefWithPrior("n_ism", 5, 400, Scale.LOG),  # mode ~147 hugged old 150 wall
        #ParamDefWithPrior("p", 2.1, 2.5, Scale.LINEAR),
        ParamDefWithPrior("p", 2.01, 2.3, Scale.LINEAR),
        ParamDefWithPrior("eps_e", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B", 0.002, 0.05, Scale.LOG),  # mode ~0.0056; let left tail close
        ParamDefWithPrior("xi", 0.8, 1.0, Scale.LINEAR),
        ParamDefWithPrior("tau", 5, 30, Scale.LOG),  # Tighter: 5-30s to prevent late RS peak

        # Reverse shock (constrained to prevent unphysical late peak)
        ParamDefWithPrior("p_r", 2.0, 3.0, Scale.LINEAR),
        ParamDefWithPrior("eps_e_r", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B_r", 0.005, 0.3, Scale.LOG),  # Lower upper limit: prevent RS dominance
        ParamDefWithPrior("xi_r", 0.7, 1.0, Scale.LINEAR),

        # Host extinction
        ParamDefWithPrior(
            "A_V", 0.001, 2.0, Scale.LOG,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]

    if include_flare:
        params.extend([
            ParamDefWithPrior("t_start_flare", 1000, 5000, Scale.LOG),  # Wider range
            ParamDefWithPrior("tau_rise_flare", 10, 2000, Scale.LOG),  # optimum 25.5s was below old 30s bound
            ParamDefWithPrior("tau_decay_flare", 1000, 10000, Scale.LOG),  # Extended
            ParamDefWithPrior("A_flare", 1e-10, 5e-9, Scale.LOG),  # Extended: allow brighter flares
            ParamDefWithPrior("flare_beta", 0.5, 1.2, Scale.LINEAR),
        ])

    if include_wing:
        params.extend([
            ParamDefWithPrior("E_iso_wing", 1e51, 1e53, Scale.LOG),  # old 1e52 floor clipped init & posterior
            ParamDefWithPrior("Gamma0_wing", 10, 100, Scale.LOG),  # Extended upper limit
            ParamDefWithPrior("theta_c_wing", 0.2, 0.7, Scale.LOG),  # mode 0.49 hugged old 0.5 wall
            ParamDefWithPrior("p_wing", 2.2, 3.3, Scale.LINEAR),  # optimum 3.06 was above old 2.9 bound
            ParamDefWithPrior("eps_e_wing", 0.1, 1.0, Scale.LOG),  # mode 0.30 hugged old 0.3 wall
            ParamDefWithPrior("eps_B_wing", 0.001, 0.02, Scale.LOG),
            ParamDefWithPrior("xi_wing", 0.6, 1.0, Scale.LINEAR),
        ])

    return params


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
            nu_xrt_min = 7.25e16
            nu_xrt_max = 2.42e18
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
        
        # Convert to mJy
        opt_flux_mJy = opt_flux * 1e26
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


def main():
    parser = argparse.ArgumentParser(description="Final model: ALL data + Core + Wing + RS + Norris flare")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-wing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-spectral-index", default=True, action=argparse.BooleanOptionalAction,
                        help="Constrain the fit with the XRT spectral index (default: on)")
    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--ncpus", type=int, default=64)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    # Load ALL data
    print("Loading ALL data (XRT + all optical bands)...")
    xrt_data, optical_datasets = load_all_optical_data()

    # XRT spectral index (photon index) constraint
    xrt_index_data = None
    if args.use_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
        except Exception as e:
            print(f"  Warning: could not load XRT spectral index ({e}); continuing without it")
    
    print(f"\nData loaded:")
    print(f"  XRT: {len(xrt_data['time'])} points ({xrt_data['time'].min()/3600:.2f}-{xrt_data['time'].max()/3600:.1f} hr)")
    for dataset in optical_datasets:
        print(f"  {dataset['name']}: {len(dataset['time'])} points " +
              f"({dataset['time'].min()/3600:.2f}-{dataset['time'].max()/3600:.1f} hr)")
    
    total_optical = sum(len(d['time']) for d in optical_datasets)
    print(f"\nTotal: {len(xrt_data['time'])} XRT + {total_optical} optical = {len(xrt_data['time']) + total_optical} points")
    print(f"Include flare: {args.include_flare}")
    print(f"Include wing: {args.include_wing}")
    if xrt_index_data is not None:
        print(f"XRT spectral index: {len(xrt_index_data['time'])} points "
              f"(applied where core+wing dominate XRT, flare < {SI_FLARE_FRAC_MAX:.0%})")
    else:
        print("XRT spectral index: not used")
    
    # Setup parameters
    param_defs = make_param_defs(include_flare=args.include_flare, include_wing=args.include_wing)
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    ndim = len(labels)

    # Determine core budget first so we can size the walker ensemble to feed it.
    # emcee evaluates walkers in two half-batches, so effective parallelism is
    # capped at nwalkers/2 -> use ~2 walkers per worker.
    n_cpus = mp.cpu_count()
    n_workers = min(args.ncpus, n_cpus - 2)  # leave headroom on the shared machine
    nwalkers = args.nwalkers or max(2 * ndim, 2 * n_workers)
    nwalkers += nwalkers % 2  # emcee requires an even number of walkers
    n_workers = min(n_workers, nwalkers // 2)  # no worker should sit idle

    print(f"\nParameters: {ndim}")
    print(f"Walkers: {nwalkers}")
    print(f"Steps: {args.nsteps}")
    
    # Initial positions: joint re-optimization of the
    # final_flare_wing_20260724_171919 best fit inside the widened bounds
    # (logL = -548.0 under the current data + spectral-index likelihood).
    # NOTE: the previous guess had p_r=3.329 and E_iso_wing=3e51 OUTSIDE their
    # own bounds, so every walker started clipped onto those walls.
    initial_guess = {
        "E_iso_core": 1.124e52,
        "Gamma0_core": 551,
        "theta_c_core": 0.0391,
        "n_ism": 146.9,
        "p": 2.164,
        "eps_e": 0.0416,
        "eps_B": 0.00563,
        "xi": 0.897,
        "tau": 12.8,
        "p_r": 2.30,
        "eps_e_r": 0.0511,
        "eps_B_r": 0.162,
        "xi_r": 0.852,
        "A_V": 0.238,
        "t_start_flare": 2553,
        "tau_rise_flare": 25.5,
        "tau_decay_flare": 2391,
        "A_flare": 9.62e-10,
        "flare_beta": 0.638,
        "E_iso_wing": 1.011e52,
        "Gamma0_wing": 19.2,
        "theta_c_wing": 0.492,
        "p_wing": 3.06,
        "eps_e_wing": 0.303,
        "eps_B_wing": 0.0121,
        "xi_wing": 0.98,  # optimum is at the physical limit 1.0; start just inside
    }
    
    pos0 = []
    for p in param_defs:
        if p.name in initial_guess:
            center = initial_guess[p.name]
            # The initial guess is a converged optimum (logL ~ -550), so scatter
            # walkers at roughly the posterior width (~0.1 dex) instead of the
            # 0.3 dex used when the guess was rough - otherwise the ensemble
            # starts ~50 logL downhill and wastes steps re-converging.
            if p.scale is Scale.LOG:
                center_log = np.log10(center)
                pos0.append(np.random.normal(center_log, 0.1, nwalkers))
            else:
                pos0.append(np.random.normal(center, center * 0.05, nwalkers))
        else:
            if p.scale is Scale.LOG:
                lower_log = np.log10(p.lower)
                upper_log = np.log10(p.upper)
                pos0.append(np.random.uniform(lower_log, upper_log, nwalkers))
            else:
                pos0.append(np.random.uniform(p.lower, p.upper, nwalkers))
    
    pos0 = np.array(pos0).T
    
    # Clip to bounds
    for i, p in enumerate(param_defs):
        if p.scale is Scale.LOG:
            lower_log = np.log10(p.lower)
            upper_log = np.log10(p.upper)
            pos0[:, i] = np.clip(pos0[:, i], lower_log, upper_log)
        else:
            pos0[:, i] = np.clip(pos0[:, i], p.lower, p.upper)
    
    # Run MCMC with multiprocessing
    print("\nRunning MCMC...")
    print(f"Using {n_workers} CPU cores (out of {n_cpus} available)")

    with Pool(n_workers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim,
            log_probability,
            args=(param_defs, xrt_data, optical_datasets, args.include_flare, args.include_wing,
                  xrt_index_data),
            pool=pool,
        )
        sampler.run_mcmc(pos0, args.nsteps, progress=True)
    
    # Save results
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = "final"
    if args.include_flare:
        phase_name += "_flare"
    if args.include_wing:
        phase_name += "_wing"
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"{phase_name}_{run_ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    
    np.save(outdir / "samples.npy", samples)
    np.save(outdir / "log_probs.npy", log_probs)
    (outdir / "labels.txt").write_text("\n".join(labels))
    
    top_params, top_log_probs = top_k_samples(samples, log_probs, 10)
    np.save(outdir / "top_k_params.npy", top_params)
    np.save(outdir / "top_k_log_probs.npy", top_log_probs)
    
    print(f"\nBest log probability: {top_log_probs[0]:.3f}")
    print(f"Results saved to: {outdir}")
    
    # Save summary
    lines = [
        "=== Fit Configuration ===",
        f"Model: Core+RS + Norris flare + Wing jet",
        f"XRT data: {len(xrt_data['time'])} points",
    ]
    for dataset in optical_datasets:
        lines.append(f"{dataset['name']}: {len(dataset['time'])} points")
    lines.append(f"Best log probability: {top_log_probs[0]:.6g}")
    lines.append("")
    lines.append("=== Best-fit Parameters ===")
    lines.append(f"{'label':<20} {'sampled':>14} {'physical':>14}")
    lines.append("-" * 50)
    
    for label, param_def, sampled in zip(labels, param_defs, top_params[0]):
        if param_def.scale is Scale.LOG:
            physical = 10 ** sampled
        else:
            physical = sampled
        lines.append(f"{label:<20} {sampled:>14.6g} {physical:>14.6g}")
    
    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")
    
    # Plot light curves with best-fit model
    print("\nPlotting best-fit light curves...")
    plot_light_curves(outdir)

    if xrt_index_data is not None:
        print("Plotting spectral index comparison...")
        plot_spectral_index_comparison(outdir)

    plot_corner(outdir, labels, max_samples=20000)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")
    
    print("\n" + "=" * 60)
    print("✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
