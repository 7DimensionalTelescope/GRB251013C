#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import os

import argparse
import bilby
import emcee
import numpy as np

from VegasAfterglow import Fitter, ISM, Model, Observer, Radiation, TophatJet, ParamDef, Scale
try:
    from VegasAfterglow.fitting.utils import _build_transformer
except ImportError:
    from VegasAfterglow.runner import _build_transformer
from VegasAfterglow.units import Hz, keV, mJy, sec

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import D_L, REDSHIFT
from grb.io import read_data
from early_phase_plotting import plot_best_fit, plot_spectral_index_comparison
from utils import (
    HOST_AV_LOG10_MEAN,
    HOST_AV_LOG10_SIGMA,
    ParamDefWithPrior,
    XRT_FLARE_START_TIME,
    compute_p_prior_from_spectral_index,
    compute_break_frequencies,
    load_xrt_spectral_index,
    plot_corner,
    xrt_flux_error,
)
from spectral_index_interpolator import get_spectral_index_calculator


I_DATA_ROWS = 40
XRT_BAND = (0.3 * keV, 10 * keV)
PROJECT_DIR = Path(__file__).absolute().parent
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


def make_fitter(xrt_data, i_data):
    fitter = Fitter(
        z=REDSHIFT,
        lumi_dist=D_L,
        jet="tophat",
        medium="ism",
        rvs_shock=True,
        fwd_ssc=False,
        rvs_ssc=False,
        kn=False,
        magnetar=False,
        rtol=1e-5,
        resolution=(0.1, 0.25, 10),
        extinction="mw",
    )

    fitter.add_flux(
        band=XRT_BAND,
        t=xrt_data["time"].to_numpy(float) * sec,
        flux=xrt_data["flux"].to_numpy(float),
        err=xrt_flux_error(xrt_data),
        num_points=10,
    )

    fitter.add_flux_density(
        nu=float(i_data["frequency_Hz"].iloc[0]) * Hz,
        t=i_data["time"].to_numpy(float) * sec,
        f_nu=i_data["flux_mJy"].to_numpy(float) * mJy,
        err=i_data["flux_mJy_error"].to_numpy(float) * mJy,
    )

    return fitter


def make_param_defs():
    return [
        ParamDefWithPrior("E_iso", 1e50, 1e55, Scale.LOG),
        ParamDefWithPrior("Gamma0", 10, 2000, Scale.LOG),  
        ParamDefWithPrior("theta_c", 1e-3, 0.1, Scale.LOG),
        ParamDefWithPrior("n_ism", 1e-4, 1e4, Scale.LOG),  
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


def param_defs_for_saved_result(plot_dir, default_param_defs):
    top_params = np.load(Path(plot_dir) / "top_k_params.npy")
    n_saved_params = top_params.shape[-1]
    if n_saved_params == len(default_param_defs):
        return default_param_defs
    return default_param_defs[:n_saved_params]


def best_params_from_result(result):
    if result.top_k_params is not None and len(result.top_k_params) > 0:
        return result.top_k_params[0]
    return result.samples[np.nanargmax(result.log_probs)]


def save_result_arrays(result, outdir):
    np.save(outdir / "samples.npy", result.samples)
    np.save(outdir / "log_probs.npy", result.log_probs)
    if result.top_k_params is not None:
        np.save(outdir / "top_k_params.npy", result.top_k_params)
    if result.top_k_log_probs is not None:
        np.save(outdir / "top_k_log_probs.npy", result.top_k_log_probs)
    (outdir / "labels.txt").write_text("\n".join(result.labels) + "\n")




def physical_param_value(param_def, sampled_value):
    if param_def.scale is Scale.LOG:
        return 10 ** sampled_value
    return sampled_value


def save_bestfit_params(outdir, param_defs, xrt_data, i_data):
    labels = [line.strip() for line in (outdir / "labels.txt").read_text().splitlines() if line.strip()]
    top_params = np.load(outdir / "top_k_params.npy")
    top_log_probs = np.load(outdir / "top_k_log_probs.npy")
    best_params = top_params[0]
    best_log_prob = top_log_probs[0]

    lines = [
        "=== Fit Configuration ===",
        "Model: ISM tophat forward shock + reverse shock",
        "VEGASafterglow: Fitter(jet='tophat', medium='ism', rvs_shock=True)",
        f"XRT data: {len(xrt_data)} points, band = 0.3-10 keV",
        f"i-band data: {len(i_data)} points, first {I_DATA_ROWS} rows after sorting",
        f"XRT selection: time < {XRT_FLARE_START_TIME:.0f} s (pre-flare only)",
        f"Host A_V prior: log10(A_V) ~ Normal({HOST_AV_LOG10_MEAN}, {HOST_AV_LOG10_SIGMA})",
        f"Best log probability: {best_log_prob:.6g}",
        "",
        "=== Best-fit Parameters ===",
        f"{'label':<18} {'sampled':>14} {'physical':>14} {'prior_min':>14} {'prior_max':>14}",
        "-" * 80,
    ]

    for label, param_def, sampled in zip(labels, param_defs, best_params):
        physical = physical_param_value(param_def, sampled)
        lines.append(
            f"{label:<18} {sampled:>14.6g} {physical:>14.6g} "
            f"{param_def.lower:>14.6g} {param_def.upper:>14.6g}"
        )

    lines.extend([
        "",
        "=== Top-k Log Probabilities ===",
    ])
    for idx, log_prob in enumerate(top_log_probs, start=1):
        lines.append(f"{idx:02d}: {log_prob:.6g}")

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fit early XRT and i-band data with ISM FS/RS VEGASafterglow model.")
    parser.add_argument("--sampler", default="emcee", help="Sampler to use (emcee, bilby, etc.).")
    parser.add_argument("--npool", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--nsteps", type=int, default=10000, help="Number of MCMC steps (default: 10000).")
    parser.add_argument("--nburn", type=int, default=2000, help="Number of burn-in steps to discard (default: 2000).")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--plot-from", default=None, help="Existing run directory with top_k_params.npy to replot.")
    parser.add_argument("--corner-samples", type=int, default=20000, help="Maximum samples to use for corner plot.")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and validate parameters.")
    parser.add_argument(
        "--use-spectral-index",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use XRT spectral index to constrain p via Gaussian prior (default: True).",
    )
    parser.add_argument(
        "--spectral-index-cooling",
        choices=("slow", "fast", "both", "smooth"),
        default="smooth",
        help="Spectral index calculation: 'slow' (p=2β+1), 'fast' (p=2β), 'both' (average), or 'smooth' (Granot & Sari 2001 - recommended).",
    )
    return parser.parse_args()


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
        xi_e=params["xi_e"],
        ssc=False,
        kn=False,
    )
    
    rvs_radiation = Radiation(
        eps_e=params["eps_e_r"],
        eps_B=params["eps_B_r"],
        p=params["p_r"],
        xi_e=params["xi_e_r"],
        ssc=False,
        kn=False,
    )
    
    return Model(
        jet=jet,
        medium=medium,
        observer=observer,
        fwd_rad=fwd_radiation,
        rvs_rad=rvs_radiation,
    ), params["A_V"]


def chi2_with_spectral_index(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Compute chi2 including XRT spectral index constraints."""
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
    from utils import host_extinction_attenuation
    attenuation = host_extinction_attenuation(i_freqs, A_V, REDSHIFT)
    i_model_flux_density_attenuated = i_model_flux_density * attenuation
    
    # XRT chi2
    xrt_diff = xrt_data["flux"].to_numpy(float) - xrt_model_flux
    chi2 = np.sum((xrt_diff / xrt_flux_error(xrt_data)) ** 2)
    
    # i-band chi2
    i_diff = i_data["flux_mJy"].to_numpy(float) - i_model_flux_density_attenuated / mJy
    chi2 += np.sum((i_diff / i_data["flux_mJy_error"].to_numpy(float)) ** 2)
    
    # Add spectral index constraint using smooth transitions
    if xrt_index_data is not None:
        # Extract physical parameters for break frequency calculations
        params_dict = {}
        for idx, param_def in enumerate(param_defs):
            value = params_array[idx]
            if param_def.scale is Scale.LOG:
                params_dict[param_def.name] = 10 ** value
            else:
                params_dict[param_def.name] = value
        
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
            chi2 += ((beta_obs - beta_model) / beta_err) ** 2
    
    return float(chi2)


def log_likelihood(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Log-likelihood function for emcee."""
    chi2 = chi2_with_spectral_index(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    return -0.5 * chi2


def log_prior(params_array, param_defs):
    """Log-prior function for emcee."""
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
    
    return log_prob


def log_probability(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Log-probability function for emcee."""
    lp = log_prior(params_array, param_defs)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(params_array, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
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


def run_custom_emcee(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, npool=4, nsteps=10000, nburn=2000):
    """Run emcee with custom likelihood including spectral index."""
    from concurrent.futures import ProcessPoolExecutor
    
    # Initial values
    ndim = len(param_defs)
    nwalkers = max(2 * ndim, 32)
    
    # Starting positions
    p0 = []
    for _ in range(nwalkers):
        pos = []
        for param_def in param_defs:
            if param_def.scale is Scale.LOG:
                lower = np.log10(param_def.lower)
                upper = np.log10(param_def.upper)
            else:
                lower = param_def.lower
                upper = param_def.upper
            
            # Start near middle with some scatter
            center = 0.5 * (lower + upper)
            width = 0.1 * (upper - lower)
            value = np.random.normal(center, width)
            value = np.clip(value, lower, upper)
            pos.append(value)
        p0.append(pos)
    
    # Setup context for multiprocessing
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    
    # Run sampler
    if npool > 1:
        with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as executor:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, _log_probability_from_context,
                pool=executor
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        _init_early_phase_context(context)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_probability_from_context
        )
        sampler.run_mcmc(p0, nsteps, progress=True)
    
    # Extract results
    samples = sampler.get_chain(discard=nburn, flat=True)
    log_probs = sampler.get_log_prob(discard=nburn, flat=True)
    
    # Get labels
    labels = []
    for param_def in param_defs:
        if param_def.scale is Scale.LOG:
            labels.append(f"log10_{param_def.name}")
        else:
            labels.append(param_def.name)
    
    # Create result object
    class Result:
        pass
    
    result = Result()
    result.samples = samples
    result.log_probs = log_probs
    result.labels = labels
    
    # Get top-k samples
    from utils import top_k_samples
    result.top_k_params, result.top_k_log_probs = top_k_samples(samples, log_probs, top_k=10)
    
    return result


def main():
    args = parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else PROJECT_DIR / "fit_results" / f"early_phase_{run_ts}"

    xrt_data, i_data, xrt_index_data = load_fit_data(include_spectral_index=args.use_spectral_index)
    fitter = make_fitter(xrt_data, i_data)
    param_defs = make_param_defs()
    vegas_param_defs = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
    fitter.validate_parameters(vegas_param_defs)

    print(f"XRT points used: {len(xrt_data)}")
    print(f"i-band points used: {len(i_data)}")
    print(f"i-band time range: {i_data['time'].min():.1f} - {i_data['time'].max():.1f} s")
    print(f"XRT time range: {xrt_data['time'].min():.1f} - {xrt_data['time'].max():.1f} s")
    if args.use_spectral_index and xrt_index_data is not None:
        print(f"XRT spectral index points: {len(xrt_index_data['time'])}")
    print(f"Sampler: {args.sampler}, npool={args.npool}")
    if args.plot_from is None:
        print(f"Output directory: {outdir}")
    else:
        print(f"Plot source directory: {args.plot_from}")

    if args.dry_run:
        print("Dry run complete: data loaded and parameters validated.")
        return

    if args.plot_from is not None:
        plot_dir = Path(args.plot_from)
        param_defs = param_defs_for_saved_result(plot_dir, param_defs)
        vegas_param_defs_plot = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
        fitter._to_params = _build_transformer(vegas_param_defs_plot)
        best_params = np.load(plot_dir / "top_k_params.npy")[0]
        plot_best_fit(fitter, best_params, xrt_data, i_data, plot_dir, XRT_BAND, HOST_AV_PARAM)
        save_bestfit_params(plot_dir, param_defs, xrt_data, i_data)
        plot_corner(plot_dir, max_samples=args.corner_samples)
        print(f"Best-fit plot saved to: {plot_dir / 'bestfit_lc.png'}")
        print(f"Best-fit parameters saved to: {plot_dir / 'bestfit_params.txt'}")
        print(f"Corner plot saved to: {plot_dir / 'corner_plot.png'}")
        return

    # Use custom emcee if spectral index constraint is enabled
    used_custom_emcee = args.use_spectral_index and xrt_index_data is not None
    
    if used_custom_emcee:
        print("Using custom emcee with per-measurement spectral index constraints...")
        result = run_custom_emcee(
            param_defs, xrt_data, i_data, xrt_index_data, 
            args.spectral_index_cooling, npool=args.npool,
            nsteps=args.nsteps, nburn=args.nburn
        )
    else:
        result = fitter.fit(
            vegas_param_defs,
            sampler=args.sampler,
            npool=args.npool,
            outdir=str(outdir),
            label="early_phase",
            top_k=10,
            priors=make_priors(param_defs, 
                              xrt_index_data if args.use_spectral_index else None,
                              args.spectral_index_cooling),
        )
    
    # Create output directory only after successful fitting
    outdir.mkdir(parents=True, exist_ok=True)
    
    save_result_arrays(result, outdir)
    best_params = best_params_from_result(result)
    best_log_prob = np.nanmax(result.log_probs)

    save_bestfit_params(outdir, param_defs, xrt_data, i_data)
    print(f"Best log probability: {best_log_prob:.3f}")
    print(f"Best-fit parameters saved to: {outdir / 'bestfit_params.txt'}")
    
    # Set up fitter for plotting (even when using custom emcee)
    if used_custom_emcee:
        # When using custom emcee, fitter wasn't fitted, so set up transformer manually
        vegas_param_defs_plot = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
        fitter._to_params = _build_transformer(vegas_param_defs_plot)
    
    plot_best_fit(fitter, best_params, xrt_data, i_data, outdir, XRT_BAND, HOST_AV_PARAM)
    print(f"Best-fit plot saved to: {outdir / 'bestfit_lc.png'}")
    
    plot_corner(outdir, max_samples=args.corner_samples)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")
    
    if args.use_spectral_index and xrt_index_data is not None:
        plot_spectral_index_comparison(
            best_params,
            result.labels,
            outdir,
            xrt_index_data
        )
    

if __name__ == "__main__":
    main()
