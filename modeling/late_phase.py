#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import argparse
import os

import numpy as np
import emcee

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet, PowerLawJet, GaussianJet
from VegasAfterglow.units import keV, mJy

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import D_L, HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA, REDSHIFT, TRIGGER_TIME, XRT_EXCLUDE_TIME_RANGE
from grb.io import filter_data, read_data
from late_phase_plotting import plot_best_fit, plot_corner, plot_spectral_index_comparison
from grb.extinction import host_extinction_attenuation
from grb.results import read_labels, top_k_samples
from grb.spectral_index import compute_break_frequencies, load_xrt_spectral_index
from grb.utils import flux_error
from spectral_index_interpolator import get_spectral_index_calculator


PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"
XRT_BAND = (0.3 * keV, 10 * keV)
XRT_LABEL = "XRT"
_LOG_PROB_CONTEXT = None


@dataclass
class Param:
    name: str
    lower: float
    upper: float
    log10: bool = True
    initial: float = None
    gaussian_prior: Optional[Tuple[float, float]] = None
    
    def has_gaussian_prior(self):
        return self.gaussian_prior is not None
    
    def get_prior_mean_sigma(self):
        if self.gaussian_prior is None:
            return None, None
        return self.gaussian_prior


@dataclass
class OpticalDataset:
    name: str
    frequency_hz: float
    time_s: np.ndarray
    flux_mjy: np.ndarray
    flux_err_mjy: np.ndarray


def seconds_from_trigger(date_obs):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.strptime(str(date_obs), fmt) - TRIGGER_TIME).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date_obs format: {date_obs}")


def load_late_phase_data(include_spectral_index=False):
    xrt_data = read_data("xrt")
    xrt_data = xrt_data[~xrt_data["time"].between(*XRT_EXCLUDE_TIME_RANGE)].reset_index(drop=True)

    optical = []

    i_data = read_data(
        "i_data",
        correct_galactic_extinction=True,
        add_converted_flux=True,
    ).copy()
    optical.append(_dataset_from_dataframe("i", i_data, time_column="time"))

    circular = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
    for filter_name in ("Rc", "Ic"):
        data = filter_data(
            circular,
            filter_name=filter_name,
            facility_name="Leavitt",
            remove_upper_limits=True,
        )
        optical.append(_dataset_from_dataframe(f"Leavitt {filter_name}", data, time_column="time"))

    sdt_data = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
    sdt_data = sdt_data[~sdt_data["is_upper_limit"].astype(bool)].copy()
    for _, row in sdt_data.iterrows():
        optical.append(
            OpticalDataset(
                name=f"7DT {row['filter_name']}",
                frequency_hz=float(row["frequency_Hz"]),
                time_s=np.array([seconds_from_trigger(row["date_obs"])]),
                flux_mjy=np.array([float(row["flux_mJy"])]),
                flux_err_mjy=np.array([float(row["flux_mJy_error"])]),
            )
        )

    xrt_index_data = None
    if include_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
            # Filter to exclude flare times
            mask = ~np.array([
                XRT_EXCLUDE_TIME_RANGE[0] <= t <= XRT_EXCLUDE_TIME_RANGE[1]
                for t in xrt_index_data["time"]
            ])
            xrt_index_data = {k: v[mask] for k, v in xrt_index_data.items()}
        except Exception as e:
            print(f"Warning: Could not load XRT spectral index data: {e}")
            xrt_index_data = None

    return xrt_data, optical, xrt_index_data


def _dataset_from_dataframe(name, df, time_column):
    return OpticalDataset(
        name=name,
        frequency_hz=float(df["frequency_Hz"].iloc[0]),
        time_s=df[time_column].to_numpy(float),
        flux_mjy=df["flux_mJy"].to_numpy(float),
        flux_err_mjy=df["flux_mJy_error"].to_numpy(float),
    )


def find_latest_early_dir():
    candidates = []
    for path in FIT_RESULTS_DIR.glob("early_phase_*"):
        if (path / "top_k_params.npy").exists() and (path / "labels.txt").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No usable early_phase_* directory with top_k_params.npy and labels.txt")
    return sorted(candidates)[-1]


def early_dir_from_late_result(late_dir):
    bestfit_file = Path(late_dir) / "bestfit_params.txt"
    if not bestfit_file.exists():
        return None

    prefixes = ("Early _w component directory:", "Early core directory:")
    for line in bestfit_file.read_text().splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                path = Path(line.removeprefix(prefix).strip())
                if (path / "top_k_params.npy").exists() and (path / "labels.txt").exists():
                    return path
    return None


def load_early_core_params(early_dir):
    early_dir = Path(early_dir)
    labels = read_labels(early_dir / "labels.txt")
    sampled = np.load(early_dir / "top_k_params.npy")[0]
    params = {}
    for label, value in zip(labels, sampled):
        if label.startswith("log10_"):
            params[label.replace("log10_", "")] = 10 ** value
        else:
            params[label] = value
    return params


# (wing param name, matching core param name, source key in early_params)
WING_SPECS = [
    ("E_iso_w", "E_iso", "E_iso"),
    ("Gamma0_w", "Gamma0", "Gamma0"),
    ("theta_c_w", "theta_c", "theta_c"),
    ("eps_e_w", "eps_e", "eps_e"),
    ("eps_B_w", "eps_B", "eps_B"),
    ("p_w", "p", "p"),
    ("xi_w", "xi", "xi_e"),
]


def early_wing_params(early_params):
    defaults = {"xi_e": 1.0}
    params = {
        wing_name: early_params.get(early_key, defaults.get(early_key))
        for wing_name, _, early_key in WING_SPECS
    }
    params["n_ism"] = early_params["n_ism"]
    
    # Reverse shock parameters (fixed from early result)
    params["tau"] = early_params.get("tau", 1.0)
    params["p_r"] = early_params.get("p_r", early_params.get("p", 2.2))
    params["eps_e_r"] = early_params.get("eps_e_r", early_params.get("eps_e", 0.1))
    params["eps_B_r"] = early_params.get("eps_B_r", early_params.get("eps_B", 0.01))
    params["xi_e_r"] = early_params.get("xi_e_r", 1.0)
    return params


def make_wing_model(wing_params, early_params=None, jet_type="tophat"):
    # For reverse shock, use fixed early params as fallback
    if early_params is None:
        early_params = {}
    
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=wing_params["n_ism"])
    
    # Create jet based on type
    if jet_type == "tophat":
        jet = TophatJet(
            E_iso=wing_params["E_iso_w"],
            Gamma0=wing_params["Gamma0_w"],
            theta_c=wing_params["theta_c_w"],
            duration=wing_params.get("tau", early_params.get("tau", 1.0)),
        )
    elif jet_type == "powerlaw":
        jet = PowerLawJet(
            E_iso=wing_params["E_iso_w"],
            Gamma0=wing_params["Gamma0_w"],
            theta_c=wing_params["theta_c_w"],
            k_e=wing_params.get("k_e_w", 2.0),
            k_g=wing_params.get("k_g_w", 2.0),
            duration=wing_params.get("tau", early_params.get("tau", 1.0)),
        )
    elif jet_type == "gaussian":
        jet = GaussianJet(
            E_iso=wing_params["E_iso_w"],
            Gamma0=wing_params["Gamma0_w"],
            theta_c=wing_params["theta_c_w"],
            duration=wing_params.get("tau", early_params.get("tau", 1.0)),
        )
    else:
        raise ValueError(f"Unknown jet type: {jet_type}")
    
    fwd_radiation = Radiation(
        eps_e=wing_params["eps_e_w"],
        eps_B=wing_params["eps_B_w"],
        p=wing_params["p_w"],
        xi_e=wing_params["xi_w"],
        ssc=False,
        kn=False,
    )
    
    # Reverse shock uses fixed early params
    rvs_radiation = Radiation(
        eps_e=wing_params.get("eps_e_r", early_params.get("eps_e_r", early_params.get("eps_e", 0.1))),
        eps_B=wing_params.get("eps_B_r", early_params.get("eps_B_r", early_params.get("eps_B", 0.01))),
        p=wing_params.get("p_r", early_params.get("p_r", early_params.get("p", 2.2))),
        xi_e=wing_params.get("xi_e_r", early_params.get("xi_e_r", 1.0)),
        ssc=False,
        kn=False,
    )
    return Model(
        jet=jet,
        medium=medium,
        observer=observer,
        fwd_rad=fwd_radiation,
        rvs_rad=rvs_radiation,
        resolutions=(0.1, 0.25, 10),
    )


def make_fixed_early_wing_model(early_params, jet_type="tophat"):
    return make_wing_model(early_wing_params(early_params), early_params, jet_type)


def make_core_model(params, early_params, jet_type="tophat"):
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params.get("n_ism", early_params["n_ism"]))
    
    # Create jet based on type
    if jet_type == "tophat":
        jet = TophatJet(
            E_iso=params["E_iso"],
            Gamma0=params["Gamma0"],
            theta_c=params["theta_c"],
        )
    elif jet_type == "powerlaw":
        jet = PowerLawJet(
            E_iso=params["E_iso"],
            Gamma0=params["Gamma0"],
            theta_c=params["theta_c"],
            k_e=params.get("k_e", 2.0),  # Energy index
            k_g=params.get("k_g", 2.0),  # Lorentz factor index
        )
    elif jet_type == "gaussian":
        jet = GaussianJet(
            E_iso=params["E_iso"],
            Gamma0=params["Gamma0"],
            theta_c=params["theta_c"],
        )
    else:
        raise ValueError(f"Unknown jet type: {jet_type}. Choose 'tophat', 'powerlaw', or 'gaussian'")
    
    radiation = Radiation(
        eps_e=params["eps_e"],
        eps_B=params["eps_B"],
        p=params["p"],
        xi_e=params["xi"],
        ssc=False,
        kn=False,
    )
    return Model(jet=jet, medium=medium, observer=observer, fwd_rad=radiation, resolutions=(0.1, 0.25, 10))


def make_param_defs(early_params, free_w=True, jet_type="tophat"):
    core_defs = [
        Param("E_iso", 1e50, 1e57, True, 3e52),
        Param("Gamma0", 10, 2000, True, 100),  # Increased from 1000 to 2000
        Param("theta_c", 1e-3, 0.5, True, 0.03),  # Increased from 0.1 to 0.5
        Param("eps_e", 1e-3, 1.0, True, 0.3),  # Increased from 0.5 to 1.0 (equipartition)
        Param("eps_B", 1e-6, 1.0, True, 0.03),  # Increased from 0.5 to 1.0 (equipartition)
        Param("p", 2.0, 3.5, False, 2.2),  # Increased from 3.0 to 3.5
        Param("xi", 1e-3, 1e2, True, 1.0),
        Param(
            "A_V",
            10 ** (HOST_AV_LOG10_MEAN - 5 * HOST_AV_LOG10_SIGMA),
            10 ** (HOST_AV_LOG10_MEAN + 5 * HOST_AV_LOG10_SIGMA),
            True,
            0.3,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]
    
    # Add structured jet parameters
    if jet_type == "powerlaw":
        core_defs.extend([
            Param("k_e", 0.0, 10.0, False, 2.0),  # Energy structure index
            Param("k_g", 0.0, 10.0, False, 2.0),  # Lorentz factor structure index
        ])
    
    if free_w:
        wing_defs = make_wing_param_defs(early_params, core_defs, jet_type)
        n_ism_def = Param("n_ism", 1e-4, 1e4, True, min(max(early_params["n_ism"], 1e-4), 1e4))  # Increased from 1e3 to 1e4
        return core_defs + wing_defs + [n_ism_def]
    return core_defs


def make_wing_param_defs(early_params, core_defs, jet_type="tophat"):
    core_by_name = {param.name: param for param in core_defs}
    wing = early_wing_params(early_params)
    wing_defs = []
    for wing_name, core_name, _ in WING_SPECS:
        core = core_by_name[core_name]
        initial = min(max(wing[wing_name], core.lower), core.upper)
        wing_defs.append(Param(wing_name, core.lower, core.upper, core.log10, initial))
    
    # Add structured jet parameters for wing component if using powerlaw
    if jet_type == "powerlaw":
        wing_defs.extend([
            Param("k_e_w", 0.0, 10.0, False, 2.0),  # Energy structure index for wing
            Param("k_g_w", 0.0, 10.0, False, 2.0),  # Lorentz factor structure index for wing
        ])
    
    return wing_defs


def sampled_labels(param_defs):
    return [f"log10_{param.name}" if param.log10 else param.name for param in param_defs]


def sampled_bounds(param_defs):
    lower = [np.log10(param.lower) if param.log10 else param.lower for param in param_defs]
    upper = [np.log10(param.upper) if param.log10 else param.upper for param in param_defs]
    return np.array(lower), np.array(upper)


def sampled_initial(param_defs):
    values = []
    for param in param_defs:
        initial = param.initial
        if initial is None:
            initial = np.sqrt(param.lower * param.upper) if param.log10 else 0.5 * (param.lower + param.upper)
        initial = min(max(initial, param.lower), param.upper)
        values.append(np.log10(initial) if param.log10 else initial)
    return np.array(values)


def late_core_initial_guesses():
    # Updated based on early_phase_20260708_145332 results
    # Early phase: E_iso=9.0e51, Gamma0=278, theta_c=0.196, eps_e=0.035, eps_B=0.176, p=2.14, xi=0.72, A_V=0.036
    base_av = 0.036  # From latest early phase fit
    
    return [
        # Guess 1: Similar to early phase but lower energy (late core should be narrower/weaker)
        {
            "E_iso": 3e52,
            "Gamma0": 100,
            "theta_c": 0.03,
            "p": 2.2,
            "eps_e": 0.05,  # Updated from 0.3
            "eps_B": 0.15,  # Updated from 0.03
            "xi": 0.7,  # Updated from 1.0
            "A_V": base_av,  # Updated from 10**(-0.82)
        },
        # Guess 2: Higher Gamma0 similar to early phase
        {
            "E_iso": 3e52,
            "Gamma0": 250,  # Updated from 150
            "theta_c": 0.03,
            "p": 2.15,  # Updated from 2.2
            "eps_e": 0.05,  # Updated
            "eps_B": 0.15,  # Updated
            "xi": 0.7,  # Updated
            "A_V": base_av,
        },
        # Guess 3: Higher energy, lower Gamma0
        {
            "E_iso": 3e53,
            "Gamma0": 50,
            "theta_c": 0.01,
            "p": 2.2,
            "eps_e": 0.05,  # Updated
            "eps_B": 0.15,  # Updated
            "xi": 0.7,  # Updated
            "A_V": base_av,
        },
        # Guess 4: Moderate energy, moderate Gamma0
        {
            "E_iso": 1e53,
            "Gamma0": 100,
            "theta_c": 0.01,
            "p": 2.2,
            "eps_e": 0.05,  # Updated
            "eps_B": 0.15,  # Updated
            "xi": 0.7,  # Updated
            "A_V": base_av,
        },
        # Guess 5: Very high energy, narrow jet
        {
            "E_iso": 1e54,
            "Gamma0": 100,
            "theta_c": 0.003,
            "p": 2.2,
            "eps_e": 0.05,  # Updated
            "eps_B": 0.15,  # Updated
            "xi": 0.7,  # Updated
            "A_V": base_av,
        },
        # Guess 6: Lower Gamma0, wider angle
        {
            "E_iso": 1e53,
            "Gamma0": 50,
            "theta_c": 0.05,  # Updated from 0.03
            "p": 2.2,
            "eps_e": 0.05,  # Updated
            "eps_B": 0.15,  # Updated
            "xi": 0.7,  # Updated
            "A_V": base_av,
        },
    ]


def sampled_from_physical(params, param_defs):
    values = []
    for param in param_defs:
        value = params.get(param.name, param.initial)
        value = min(max(value, param.lower), param.upper)
        values.append(np.log10(value) if param.log10 else value)
    return np.array(values)


def to_physical(theta, param_defs):
    return {
        param.name: 10 ** value if param.log10 else value
        for param, value in zip(param_defs, theta)
    }


def flux_predictions(model, xrt_data, optical_data):
    preds = {
        XRT_LABEL: np.asarray(
            model.flux(xrt_data["time"].to_numpy(float), XRT_BAND[0], XRT_BAND[1], 10).total
        )
    }
    for dataset in optical_data:
        preds[dataset.name] = np.asarray(
            model.flux_density(dataset.time_s, dataset.frequency_hz * np.ones_like(dataset.time_s)).total
        )
    return preds


def precompute_fixed_predictions(fixed_model, xrt_data, optical_data):
    return flux_predictions(fixed_model, xrt_data, optical_data)


def chi2_for_params(theta, param_defs, early_params, fixed_predictions, xrt_data, optical_data, free_w, 
                     xrt_index_data=None, spectral_index_cooling="slow", jet_type="tophat"):
    params = to_physical(theta, param_defs)

    fitted_core_model = make_core_model(params, early_params, jet_type=jet_type)

    if free_w:
        wing_model = make_wing_model(params, early_params, jet_type=jet_type)
        wing_predictions = flux_predictions(wing_model, xrt_data, optical_data)
    else:
        wing_predictions = fixed_predictions

    chi2 = 0.0

    xrt_model = wing_predictions[XRT_LABEL] + np.asarray(
        fitted_core_model.flux(xrt_data["time"].to_numpy(float), XRT_BAND[0], XRT_BAND[1], 10).total
    )
    xrt_diff = xrt_data["flux"].to_numpy(float) - xrt_model
    chi2 += np.sum((xrt_diff / flux_error(xrt_data)) ** 2)

    for dataset in optical_data:
        fitted_core_flux = np.asarray(
            fitted_core_model.flux_density(dataset.time_s, dataset.frequency_hz * np.ones_like(dataset.time_s)).total
        )
        attenuation = host_extinction_attenuation(
            dataset.frequency_hz * np.ones_like(dataset.time_s),
            params["A_V"],
            REDSHIFT,
        )
        model_flux_mjy = (wing_predictions[dataset.name] + fitted_core_flux) * attenuation / mJy
        chi2 += np.sum(((dataset.flux_mjy - model_flux_mjy) / dataset.flux_err_mjy) ** 2)

    # Add spectral index constraint
    if xrt_index_data is not None:
        p_core = params["p"]
        
        # Get spectral index calculator for smooth transitions
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
                breaks = compute_break_frequencies(params, REDSHIFT, t_obs)
                
                # Use smooth spectral index calculation
                beta_model = calc.beta_at_frequency(
                    nu_xrt, breaks["nu_m"], breaks["nu_c"], p_core
                )
            else:
                # Use simple step function (backward compatibility)
                if spectral_index_cooling == "slow":
                    # G&S Table 2: slow cooling between breaks
                    beta_model = (1.0 - p_core) / 2.0
                elif spectral_index_cooling == "fast":
                    # G&S Table 2: fast cooling above nu_c
                    beta_model = -p_core / 2.0
                elif spectral_index_cooling == "both":
                    # Average of both regimes
                    beta_slow = (1.0 - p_core) / 2.0
                    beta_fast = -p_core / 2.0
                    beta_model = 0.5 * (beta_slow + beta_fast)
                else:
                    raise ValueError(f"Unknown cooling regime: {spectral_index_cooling}")
            
            # Use appropriate error (asymmetric errors)
            beta_err = beta_err_high if beta_model > beta_obs else beta_err_low
            chi2 += ((beta_obs - beta_model) / beta_err) ** 2

    return float(chi2)


def log_prior(theta, param_defs, lower, upper):
    if not np.all((theta >= lower) & (theta <= upper)):
        return -np.inf

    lp = 0.0
    labels = sampled_labels(param_defs)
    params = dict(zip(labels, theta))
    
    for param_def in param_defs:
        if not param_def.has_gaussian_prior():
            continue
            
        param_name = param_def.name
        sampled_label = f"log10_{param_name}" if param_def.log10 else param_name
            
        sampled_value = params.get(sampled_label)
        if sampled_value is None:
            continue
            
        mean, sigma = param_def.get_prior_mean_sigma()
        lp += -0.5 * ((sampled_value - mean) / sigma) ** 2
        lp -= np.log(sigma * np.sqrt(2 * np.pi))
    
    return lp


def log_probability(theta, param_defs, lower, upper, early_params, fixed_predictions, xrt_data, optical_data, free_w,
                     xrt_index_data=None, spectral_index_cooling="slow", jet_type="tophat"):
    lp = log_prior(theta, param_defs, lower, upper)
    if not np.isfinite(lp):
        return -np.inf
    try:
        chi2 = chi2_for_params(theta, param_defs, early_params, fixed_predictions, xrt_data, optical_data, free_w,
                                xrt_index_data, spectral_index_cooling, jet_type)
    except Exception:
        return -np.inf
    if not np.isfinite(chi2):
        return -np.inf
    return lp - 0.5 * chi2


def _init_log_probability_context(context):
    global _LOG_PROB_CONTEXT
    _LOG_PROB_CONTEXT = context


def _log_probability_from_context(theta):
    return log_probability(theta, *_LOG_PROB_CONTEXT)


def run_emcee(param_defs, early_params, fixed_predictions, xrt_data, optical_data, args, 
               xrt_index_data=None):
    labels = sampled_labels(param_defs)
    lower, upper = sampled_bounds(param_defs)
    ndim = len(labels)
    # Use 2.5*ndim for speed (Quick Win setting) instead of default 4*ndim
    nwalkers = args.nwalkers or max(int(2.5 * ndim), 32)
    initial_guesses = [sampled_from_physical(params, param_defs) for params in late_core_initial_guesses()]
    initial_guesses.append(sampled_initial(param_defs))
    initial_guesses = np.array(initial_guesses)
    spread = args.initial_spread * (upper - lower)
    pos0 = np.empty((nwalkers, ndim))
    for idx in range(nwalkers):
        center = initial_guesses[idx % len(initial_guesses)]
        pos0[idx] = center + spread * np.random.randn(ndim)
    eps = 1e-6 * (upper - lower)
    pos0 = np.clip(pos0, lower + eps, upper - eps)

    context = (param_defs, lower, upper, early_params, fixed_predictions, xrt_data, optical_data, args.free_w,
               xrt_index_data, args.spectral_index_cooling, args.jet_type)
    if args.pool == "none" or args.npool <= 1:
        _init_log_probability_context(context)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_probability_from_context,
        )
        sampler.run_mcmc(pos0, args.nsteps, progress=True)
    else:
        executor_cls = ProcessPoolExecutor if args.pool == "process" else ThreadPoolExecutor
        kwargs = {}
        if args.pool == "process":
            kwargs = {
                "initializer": _init_log_probability_context,
                "initargs": (context,),
            }
        else:
            _init_log_probability_context(context)

        with executor_cls(max_workers=args.npool, **kwargs) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                _log_probability_from_context,
                pool=pool,
            )
            sampler.run_mcmc(pos0, args.nsteps, progress=True)

    samples = sampler.get_chain(discard=args.nburn, thin=args.thin, flat=True)
    log_probs = sampler.get_log_prob(discard=args.nburn, thin=args.thin, flat=True)
    return samples, log_probs, labels


def save_arrays(outdir, samples, log_probs, labels, top_params, top_log_probs):
    np.save(outdir / "samples.npy", samples)
    np.save(outdir / "log_probs.npy", log_probs)
    np.save(outdir / "top_k_params.npy", top_params)
    np.save(outdir / "top_k_log_probs.npy", top_log_probs)
    (outdir / "labels.txt").write_text("\n".join(labels) + "\n")


def ensure_saved_labels_match(result_dir, expected_labels):
    saved_labels = read_labels(Path(result_dir) / "labels.txt")
    if saved_labels != list(expected_labels):
        raise ValueError(
            "Saved late-phase result uses a different parameter set. "
            "Run a new late-phase fit for the fixed early _w + fitted core strategy."
        )


def save_bestfit_params(outdir, theta, log_prob, param_defs, early_params, early_dir, xrt_data, optical_data, free_w=False, jet_type="tophat"):
    jet_name = {"tophat": "TophatJet", "powerlaw": "PowerLawJet", "gaussian": "GaussianJet"}.get(jet_type, jet_type)
    lines = [
        "=== Fit Configuration ===",
        f"Model: free _w {jet_name} + fitted core {jet_name}" if free_w else f"Model: fixed early TophatJet as _w component + fitted core {jet_name}",
        f"Early _w component directory: {early_dir}",
        f"XRT data: {len(xrt_data)} points, flare excluded {XRT_EXCLUDE_TIME_RANGE[0]:.0f}-{XRT_EXCLUDE_TIME_RANGE[1]:.0f} s",
        f"Optical datasets: {sum(len(dataset.time_s) for dataset in optical_data)} points",
        f"Host A_V prior: log10(A_V) ~ Normal({HOST_AV_LOG10_MEAN}, {HOST_AV_LOG10_SIGMA})",
        f"Best log probability: {log_prob:.6g}",
    ]

    if free_w:
        lines.extend([
            "",
            "=== _w Component (TophatJet): free (fitted); n_ism shared free between _w and core ===",
            "=== Reverse shock parameters (fixed from early result) ===",
        ])
        rvs_fixed = {
            "tau": early_params.get("tau", 1.0),
            "p_r": early_params.get("p_r", early_params.get("p", 2.2)),
            "eps_e_r": early_params.get("eps_e_r", early_params.get("eps_e", 0.1)),
            "eps_B_r": early_params.get("eps_B_r", early_params.get("eps_B", 0.01)),
            "xi_e_r": early_params.get("xi_e_r", 1.0),
        }
        for key, value in rvs_fixed.items():
            lines.append(f"{key:<16} {value:.6g}")
    else:
        lines.extend([
            "",
            "=== Fixed Early Parameters Propagated to _w Component (TophatJet) ===",
        ])
        propagated = {
            "E_iso_w": early_params["E_iso"],
            "Gamma0_w": early_params["Gamma0"],
            "theta_c_w": early_params["theta_c"],
            "eps_e_w": early_params["eps_e"],
            "eps_B_w": early_params["eps_B"],
            "p_w": early_params["p"],
            "xi_w": early_params.get("xi_e", 1.0),
            "n_ism": early_params["n_ism"],
        }
        for key, value in propagated.items():
            lines.append(f"{key:<16} {value:.6g}")
        lines.extend([
            "",
            "=== Reverse shock parameters (fixed from early result) ===",
        ])
        rvs_fixed = {
            "tau": early_params.get("tau", 1.0),
            "p_r": early_params.get("p_r", early_params.get("p", 2.2)),
            "eps_e_r": early_params.get("eps_e_r", early_params.get("eps_e", 0.1)),
            "eps_B_r": early_params.get("eps_B_r", early_params.get("eps_B", 0.01)),
            "xi_e_r": early_params.get("xi_e_r", 1.0),
        }
        for key, value in rvs_fixed.items():
            lines.append(f"{key:<16} {value:.6g}")

    lines.extend([
        "",
        "=== Late Parameter Initial Values ===",
    ])
    for param in param_defs:
        lines.append(f"{param.name:<16} {param.initial:.6g}")

    lines.extend([
        "",
        "=== Late Core Initial Guess Set ===",
    ])
    for idx, guess in enumerate(late_core_initial_guesses(), start=1):
        values = ", ".join(f"{key}={value:.6g}" for key, value in guess.items())
        lines.append(f"{idx:02d}: {values}")

    lines.extend([
        "",
        "=== Best-fit New Core Parameters ===",
        f"{'label':<18} {'sampled':>14} {'physical':>14} {'prior_min':>14} {'prior_max':>14}",
        "-" * 80,
    ])
    for param, sampled in zip(param_defs, theta):
        label = f"log10_{param.name}" if param.log10 else param.name
        physical = 10 ** sampled if param.log10 else sampled
        lines.append(
            f"{label:<18} {sampled:>14.6g} {physical:>14.6g} "
            f"{param.lower:>14.6g} {param.upper:>14.6g}"
        )

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Late-phase fixed early _w component + fitted new core.")
    parser.add_argument("--early-dir", default=None, help="Early phase result directory to propagate into fixed _w component.")
    parser.add_argument(
        "--free-w",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Make the early _w component parameters free (fitted) instead of fixed from the early result (default: True).",
    )
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--nsteps", type=int, default=5000)
    parser.add_argument("--nburn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--nwalkers", type=int, default=None, help="emcee walker count; default is max(2.5 * ndim, 32) for speed.")
    parser.add_argument("--npool", type=int, default=8)  # Updated from 4 to 8 for faster fitting
    parser.add_argument(
        "--initial-spread",
        type=float,
        default=0.08,
        help="Initial walker spread as a fraction of each prior width.",
    )
    parser.add_argument(
        "--pool",
        choices=("process", "thread", "none"),
        default="process",
        help="Parallel backend for emcee likelihood calls.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corner-samples", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-from", default=None, help="Existing late_phase result directory to replot.")
    parser.add_argument(
        "--use-spectral-index",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include XRT spectral index measurements in the fit (default: True).",
    )
    parser.add_argument(
        "--spectral-index-cooling",
        choices=("slow", "fast", "both", "smooth"),
        default="smooth",
        help="Cooling regime for spectral index: 'slow' (β=(1-p)/2), 'fast' (β=-p/2), 'both' (average), or 'smooth' (Granot & Sari 2001 smooth transitions).",
    )
    parser.add_argument(
        "--jet-type",
        choices=("tophat", "powerlaw", "gaussian"),
        default="powerlaw",
        help="Jet structure type for core component: 'tophat' (uniform), 'powerlaw' (E,Gamma ∝ θ^-k), or 'gaussian' (Gaussian structure).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.early_dir:
        early_dir = Path(args.early_dir)
    elif args.plot_from:
        early_dir = early_dir_from_late_result(args.plot_from) or find_latest_early_dir()
    else:
        early_dir = find_latest_early_dir()
    early_params = load_early_core_params(early_dir)
    free_w = args.free_w
    if args.plot_from is not None:
        saved_labels = read_labels(Path(args.plot_from) / "labels.txt")
        free_w = any(label.endswith("_w") for label in saved_labels)

    xrt_data, optical_data, xrt_index_data = load_late_phase_data(include_spectral_index=args.use_spectral_index)
    fixed_model = make_fixed_early_wing_model(early_params, jet_type=args.jet_type)
    fixed_predictions = None if free_w else precompute_fixed_predictions(fixed_model, xrt_data, optical_data)
    param_defs = make_param_defs(early_params, free_w, jet_type=args.jet_type)
    labels = sampled_labels(param_defs)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"late_phase_{run_ts}"

    print(f"Early _w component directory: {early_dir}")
    print(f"_w component: {'free (fitted)' if free_w else 'fixed from early result'}")
    print(f"XRT points used: {len(xrt_data)}")
    print(f"Optical datasets: {len(optical_data)}")
    print(f"Optical points used: {sum(len(dataset.time_s) for dataset in optical_data)}")
    if args.use_spectral_index and xrt_index_data is not None:
        print(f"XRT spectral index points: {len(xrt_index_data['time'])}")
        print(f"Spectral index cooling regime: {args.spectral_index_cooling}")
    print(f"emcee pool backend: {args.pool}, npool={args.npool}")
    if args.plot_from is None:
        print(f"Output directory: {outdir}")
    else:
        print(f"Plot source directory: {args.plot_from}")

    if args.dry_run:
        print("Dry run complete: data loaded and fixed early _w component built.")
        return

    if args.plot_from is not None:
        plot_dir = Path(args.plot_from)
        ensure_saved_labels_match(plot_dir, labels)
        theta = np.load(plot_dir / "top_k_params.npy")[0]
        log_prob = np.load(plot_dir / "top_k_log_probs.npy")[0]
        wing_model = make_wing_model(to_physical(theta, param_defs), early_params, jet_type=args.jet_type) if free_w else fixed_model
        plot_best_fit(
            plot_dir,
            theta,
            param_defs,
            early_params,
            wing_model,
            xrt_data,
            optical_data,
            make_core_model,
            to_physical,
            XRT_BAND,
            flux_error,
        )
        save_bestfit_params(plot_dir, theta, log_prob, param_defs, early_params, early_dir, xrt_data, optical_data, free_w, jet_type=args.jet_type)
        
        # Generate spectral index comparison plot if data is available
        if xrt_index_data is not None and len(xrt_index_data.get("time", [])) > 0:
            plot_spectral_index_comparison(theta, labels, plot_dir, xrt_index_data)
        
        plot_corner(plot_dir, labels, max_samples=args.corner_samples)
        print(f"Best-fit plot saved to: {plot_dir / 'bestfit_lc.png'}")
        return

    samples, log_probs, labels = run_emcee(param_defs, early_params, fixed_predictions, xrt_data, optical_data, args,
                                           xrt_index_data if args.use_spectral_index else None)
    top_params, top_log_probs = top_k_samples(samples, log_probs, args.top_k)
    
    # Create output directory only after successful fitting
    outdir.mkdir(parents=True, exist_ok=True)
    
    save_arrays(outdir, samples, log_probs, labels, top_params, top_log_probs)

    best_theta = top_params[0]
    best_log_prob = top_log_probs[0]
    wing_model = make_wing_model(to_physical(best_theta, param_defs), early_params, jet_type=args.jet_type) if free_w else fixed_model
    plot_best_fit(
        outdir,
        best_theta,
        param_defs,
        early_params,
        wing_model,
        xrt_data,
        optical_data,
        make_core_model,
        to_physical,
        XRT_BAND,
        flux_error,
    )
    save_bestfit_params(outdir, best_theta, best_log_prob, param_defs, early_params, early_dir, xrt_data, optical_data, free_w, jet_type=args.jet_type)
    
    # Generate spectral index comparison plot if data is available
    if xrt_index_data is not None and len(xrt_index_data.get("time", [])) > 0:
        plot_spectral_index_comparison(best_theta, labels, outdir, xrt_index_data)
    
    plot_corner(outdir, labels, max_samples=args.corner_samples)
    print(f"Best log probability: {best_log_prob:.3f}")
    print(f"Best-fit plot saved to: {outdir / 'bestfit_lc.png'}")
    print(f"Best-fit parameters saved to: {outdir / 'bestfit_params.txt'}")
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")


if __name__ == "__main__":
    main()
