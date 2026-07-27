#!/usr/bin/env python3
"""
Partial Data Fitting - Core + Flare + Wing Model
Fit full XRT + i-band dataset with composite model
"""
from pathlib import Path
from datetime import datetime
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import emcee

from VegasAfterglow import ParamDef, Scale
from VegasAfterglow.units import keV

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA, REDSHIFT, TRIGGER_TIME
from grb.io import read_data
from grb.extinction import host_extinction_attenuation
from grb.functions import norris_flare
from grb.likelihood import log_prior
from grb.modeling import make_core_model, make_wing_model
from grb.params import ParamDefWithPrior
from grb.plotting import plot_corner
from grb.results import top_k_samples
from grb.spectral_index import compute_break_frequencies, load_xrt_spectral_index
from grb.utils import model_array
from spectral_index_interpolator import get_spectral_index_calculator

PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"
XRT_BAND = (0.3 * keV, 10.0 * keV)  # 0.3-10 keV in Hz units
I_BAND_FREQ = 3.931704e14  # Hz - actual i-band frequency from data (763 nm)
FLUX_CONVERSION_FACTOR = 1e26  # erg/s/cm²/Hz to mJy (1 mJy = 1e-26 erg/s/cm²/Hz)

# Flare time range
FLARE_TIME_RANGE = (3000, 10000)  # seconds


def make_param_defs(include_flare=True, include_wing=True):
    """Define parameters for core + flare + wing model
    
    Bounds based on successful early_phase and late_phase fits:
    - early: E_iso~9e51, Gamma0~278, theta_c~0.2, n_ism~5, p~2.1, A_V~0.04
    - late core: E_iso~4e52, Gamma0~41, theta_c~0.04, n_ism~0.9, p~2.2, A_V~0.02
    """
    params = [
        # Core jet parameters (based on early_phase best fit)
        ParamDefWithPrior("E_iso_core", 5e51, 5e52, Scale.LOG),  # Around 1e52 (TIGHTENED!)
        ParamDefWithPrior("Gamma0_core", 300, 800, Scale.LOG),   # HIGH Gamma for strong RS (TIGHTENED!)
        ParamDefWithPrior("theta_c_core", 0.02, 0.06, Scale.LOG), # Narrow ~1-3 deg (TIGHTENED!)
        
        # Shared microphysics (based on early_phase best fit)
        ParamDefWithPrior("n_ism", 10, 50, Scale.LOG),           # Around 18.76 (TIGHTENED!)
        ParamDefWithPrior("p", 2.1, 2.3, Scale.LINEAR),          # Around 2.158 (TIGHTENED!)
        ParamDefWithPrior("eps_e", 0.02, 0.1, Scale.LOG),        # Around 0.0435 (TIGHTENED!)
        ParamDefWithPrior("eps_B", 0.005, 0.05, Scale.LOG),      # LOW for strong RS! (TIGHTENED!)
        ParamDefWithPrior("xi", 0.8, 1.0, Scale.LINEAR),         # High fraction (TIGHTENED!)
        ParamDefWithPrior("tau", 10, 100, Scale.LOG),            # Around 27.6 (TIGHTENED!)
        
        # Reverse shock parameters (based on early_phase best fit)
        ParamDefWithPrior("p_r", 3.0, 3.6, Scale.LINEAR),        # Around 3.329 (TIGHTENED!)
        ParamDefWithPrior("eps_e_r", 0.02, 0.1, Scale.LOG),      # Around 0.0422 (TIGHTENED!)
        ParamDefWithPrior("eps_B_r", 0.1, 0.5, Scale.LOG),       # Around 0.246 (TIGHTENED!)
        ParamDefWithPrior("xi_r", 0.7, 1.0, Scale.LINEAR),       # Around 0.849 (TIGHTENED!)
        
        # Host extinction (much lower bound based on actual fits!)
        ParamDefWithPrior(
            "A_V",
            0.001,  # Lower bound
            2.0,    # Upper bound (was 16.98, way too high!)
            Scale.LOG,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]
    
    if include_flare:
        params.extend([
            # Flare with Norris function (fast rise + exponential decay)
            # Flare expected to peak ~1-1.5 hr based on data
            ParamDefWithPrior("t_start_flare", 2000, 4000, Scale.LOG),   # Flare start: 0.56-1.11 hr
            ParamDefWithPrior("tau_rise_flare", 100, 1000, Scale.LOG),   # Rise timescale (fast)
            ParamDefWithPrior("tau_decay_flare", 1000, 5000, Scale.LOG), # Decay timescale (slow)
            ParamDefWithPrior("A_flare", 1e-10, 5e-10, Scale.LOG),       # Flare amplitude
            ParamDefWithPrior("flare_beta", 0.6, 1.0, Scale.LINEAR),     # Flare spectral index
        ])
    
    if include_wing:
        params.extend([
            # Wide jet component - jet structure (MUCH SLOWER than core!)
            ParamDefWithPrior("E_iso_wing", 1e51, 1e52, Scale.LOG),  # Moderate energy (TIGHTENED!)
            ParamDefWithPrior("Gamma0_wing", 10, 50, Scale.LOG),     # SLOW! (TIGHTENED! was 50-500)
            ParamDefWithPrior("theta_c_wing", 0.15, 0.3, Scale.LOG), # Wide ~10-17 deg (TIGHTENED! was 0.1-0.5)
            # Wide jet component - separate microphysics (can differ from core)
            ParamDefWithPrior("p_wing", 2.2, 2.5, Scale.LINEAR),     # Slightly different (TIGHTENED!)
            ParamDefWithPrior("eps_e_wing", 0.5, 1.0, Scale.LOG),    # High (TIGHTENED!)
            ParamDefWithPrior("eps_B_wing", 1e-3, 0.01, Scale.LOG),  # Low (TIGHTENED!)
            ParamDefWithPrior("xi_wing", 0.6, 1.0, Scale.LINEAR),    # High fraction (TIGHTENED!)
        ])
    
    return params


def compute_model_components(t_array, nu, params, include_flare=True, include_wing=False):
    """Compute flux components separately
    
    Args:
        t_array: Time array in seconds (observer frame)
        nu: Either tuple (nu_min, nu_max) for energy band or scalar frequency in Hz
        params: Dictionary of physical parameters
        include_flare: Include Gaussian flare component
        include_wing: Include wide wing jet component
    
    Returns:
        Dictionary with keys: 'core', 'reverse', 'flare', 'wing', 'total'
        Each value is an array of flux values (same length as t_array)
    """
    components = {}
    
    # Core contribution (forward shock + reverse shock)
    core_model = make_core_model(params)
    if hasattr(nu, '__len__'):
        # Energy band (e.g., XRT)
        core_output = core_model.flux(t_array, nu[0], nu[1], 10)
        core_flux = model_array(core_output.total)
        core_fwd = model_array(core_output.fwd)
        core_rvs = model_array(core_output.rvs)
        is_xray = True
        # Use geometric mean frequency for X-ray band
        nu_eff = np.sqrt(nu[0] * nu[1])
    else:
        # Single frequency (e.g., i-band) - broadcast frequency to match time array
        nu_array = nu * np.ones_like(t_array)
        core_output = core_model.flux_density(t_array, nu_array)
        core_flux = model_array(core_output.total)
        core_fwd = model_array(core_output.fwd)
        core_rvs = model_array(core_output.rvs)
        is_xray = False
        nu_eff = nu
    
    components['core'] = core_fwd
    components['reverse'] = core_rvs
    total_flux = core_flux.copy()  # Total includes both FS and RS
    
    # Flare contribution with spectral scaling
    # Flare has power-law spectrum F_ν = K * ν^(-β_flare)
    # A_flare is defined as integrated flux in XRT band
    if include_flare and "t_start_flare" in params:
        # XRT band: 0.3-10 keV -> 7.25e16 - 2.42e18 Hz
        nu_xrt_min = 7.25e16  # Hz (0.3 keV)
        nu_xrt_max = 2.42e18  # Hz (10 keV)
        
        # Flare spectral index: use independent flare_beta if available,
        # otherwise fall back to synchrotron β = (p-1)/2
        if "flare_beta" in params:
            beta_flare = params["flare_beta"]
        else:
            beta_flare = (params["p"] - 1) / 2
        
        # Temporal profile: Norris function (fast rise + exponential decay)
        flare_temporal = norris_flare(
            t_array, 
            params["t_start_flare"],    # Flare start time
            params["tau_rise_flare"],   # Rise timescale
            params["tau_decay_flare"],  # Decay timescale
            params["A_flare"]
        )
        
        if is_xray:
            # For XRT, use directly (integrated flux)
            flare_factor = flare_temporal
        else:
            # For optical, convert integrated XRT flux to flux density
            # For F_ν = K * ν^(-β):
            # F_integrated = ∫_{ν1}^{ν2} K * ν^(-β) dν
            #              = K/(1-β) * (ν2^(1-β) - ν1^(1-β))  for β ≠ 1
            # Solving for K:
            # K = F_integrated * (1-β) / (ν2^(1-β) - ν1^(1-β))
            # Then F_ν(ν) = K * ν^(-β)
            
            if abs(beta_flare - 1.0) < 0.01:
                # Special case β ≈ 1
                K = flare_temporal / np.log(nu_xrt_max / nu_xrt_min)
            else:
                # General case
                K = flare_temporal * (1 - beta_flare) / (nu_xrt_max**(1-beta_flare) - nu_xrt_min**(1-beta_flare))
            
            # Flux density at optical frequency
            F_nu_optical = K * nu_eff**(-beta_flare)
            flare_factor = F_nu_optical
        
        components['flare'] = flare_factor
        total_flux = total_flux + flare_factor
    else:
        components['flare'] = np.zeros_like(t_array)
    
    # Wing contribution  
    if include_wing and "E_iso_wing" in params:
        wing_model = make_wing_model(params, spreading=False)
        if hasattr(nu, '__len__'):
            wing_flux = np.asarray(wing_model.flux(t_array, nu[0], nu[1], 10).total)
        else:
            nu_array = nu * np.ones_like(t_array)
            wing_flux = np.asarray(wing_model.flux_density(t_array, nu_array).total)
        components['wing'] = wing_flux
        total_flux = total_flux + wing_flux
    else:
        components['wing'] = np.zeros_like(t_array)
    
    components['total'] = total_flux
    return components


def compute_model_flux(t_array, nu, params, include_flare=True, include_wing=False):
    """Compute total flux from all components
    
    Args:
        t_array: Time array in seconds (observer frame)
        nu: Either tuple (nu_min, nu_max) for energy band or scalar frequency in Hz
        params: Dictionary of physical parameters
        include_flare: Include Gaussian flare component
        include_wing: Include wide wing jet component
    
    Returns:
        Array of flux values (same length as t_array)
    """
    components = compute_model_components(t_array, nu, params, include_flare, include_wing)
    return components['total']


def log_likelihood(theta, param_defs, xrt_data, i_data, include_flare, include_wing):
    """Compute log likelihood"""
    # Convert to physical parameters
    params = {}
    for param_def, value in zip(param_defs, theta):
        if param_def.scale is Scale.LOG:
            params[param_def.name] = 10 ** value
        else:
            params[param_def.name] = value
    
    try:
        chi2 = 0.0
        
        # XRT data (vectorized computation)
        xrt_times = xrt_data["time"].to_numpy(float)
        xrt_flux = xrt_data["flux"].to_numpy(float)
        xrt_flux_high = xrt_data["flux_high"].to_numpy(float)
        xrt_flux_low = xrt_data["flux_low"].to_numpy(float)
        
        xrt_model = compute_model_flux(xrt_times, XRT_BAND, params, include_flare, include_wing)
        
        # Check for invalid model values
        if not np.all(np.isfinite(xrt_model)):
            return -np.inf
        
        xrt_residuals = xrt_flux - xrt_model
        
        # Use asymmetric errors
        xrt_errors = np.where(xrt_residuals > 0, xrt_flux_high, xrt_flux_low)
        chi2 += np.sum((xrt_residuals / xrt_errors) ** 2)
        
        # i-band data (vectorized computation)
        i_times = i_data["time"].to_numpy(float)
        i_flux = i_data["flux_mJy"].to_numpy(float)
        i_flux_err = i_data["flux_mJy_error"].to_numpy(float)
        
        i_model = compute_model_flux(i_times, I_BAND_FREQ, params, include_flare, include_wing)
        
        # Apply host extinction to optical (make copy to avoid read-only array error)
        if "A_V" in params:
            attenuation = host_extinction_attenuation(I_BAND_FREQ, params["A_V"], REDSHIFT)
            i_model = i_model * attenuation
        
        # Convert to mJy
        i_model_mJy = i_model * FLUX_CONVERSION_FACTOR
        
        # Check for invalid or zero model values (common at early times for optical)
        if not np.all(np.isfinite(i_model_mJy)):
            return -np.inf
        
        # For very small model values (< 1e-10 mJy), use a floor to avoid -inf chi2
        # This handles cases where the model legitimately predicts very low optical flux
        MIN_FLUX = 1e-10  # mJy
        i_model_mJy = np.maximum(i_model_mJy, MIN_FLUX)
        
        chi2 += np.sum(((i_flux - i_model_mJy) / i_flux_err) ** 2)
        
        # Check for NaN or inf in chi2
        if not np.isfinite(chi2):
            return -np.inf
        
        return -0.5 * chi2
        
    except (ValueError, RuntimeError, ZeroDivisionError) as e:
        return -np.inf


def log_probability(theta, param_defs, xrt_data, i_data, include_flare, include_wing):
    """Combined log probability"""
    lp = log_prior(theta, param_defs)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, param_defs, xrt_data, i_data, include_flare, include_wing)
    return lp + ll


def main():
    parser = argparse.ArgumentParser(description="Partial data fitting: Core + Flare + Wing")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-wing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--nsteps", type=int, default=5000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--npool", type=int, default=8)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    
    # Load ALL data (no flare exclusion)
    print("Loading full dataset...")
    xrt_data = read_data("xrt")
    i_data = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    
    print(f"i-band frequency: {I_BAND_FREQ:.3e} Hz ({3e8/I_BAND_FREQ*1e9:.1f} nm)")
    print(f"XRT points: {len(xrt_data)} ({xrt_data['time'].min()/3600:.2f} - {xrt_data['time'].max()/3600:.1f} hr)")
    print(f"i-band points: {len(i_data)} ({i_data['time'].min()/3600:.2f} - {i_data['time'].max()/3600:.1f} hr)")
    print(f"Include flare: {args.include_flare}")
    print(f"Include wing: {args.include_wing}")
    
    # Setup parameters
    param_defs = make_param_defs(include_flare=args.include_flare, include_wing=args.include_wing)
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    ndim = len(labels)
    nwalkers = args.nwalkers or max(int(2.5 * ndim), 32)
    
    print(f"\nParameters: {ndim}")
    print(f"Walkers: {nwalkers}")
    print(f"Steps: {args.nsteps}")
    
    # Initial positions - use early_phase best-fit values as starting point
    # These parameters give excellent fit to early data with strong reverse shock
    # Early phase best fit: E_iso~1.19e52, Gamma0~522, theta_c~0.0345, n_ism~18.76
    # Forward shock: p~2.158, eps_e~0.0435, eps_B~0.0163, xi~0.943, tau~27.6
    # Reverse shock: p_r~3.329, eps_e_r~0.0422, eps_B_r~0.246, xi_r~0.849
    initial_guess = {
        # Core jet - from early_phase best fit (excellent for early data + RS)
        "E_iso_core": 1.189e52,  # From early_phase best fit
        "Gamma0_core": 522,      # From early_phase best fit - HIGH for strong RS!
        "theta_c_core": 0.0345,  # From early_phase best fit
        "n_ism": 18.76,          # From early_phase best fit
        "p": 2.158,              # From early_phase best fit
        "eps_e": 0.0435,         # From early_phase best fit
        "eps_B": 0.0163,         # From early_phase best fit - LOW for strong RS
        "xi": 0.943,             # From early_phase best fit
        "tau": 27.6,             # From early_phase best fit - jet duration
        # Reverse shock (from early_phase best fit)
        "p_r": 3.329,            # From early_phase best fit
        "eps_e_r": 0.0422,       # From early_phase best fit
        "eps_B_r": 0.246,        # From early_phase best fit
        "xi_r": 0.849,           # From early_phase best fit
        "A_V": 0.0254,           # From early_phase best fit
        # Norris flare parameters
        "t_start_flare": 3000,   # Flare starts ~0.83 hr
        "tau_rise_flare": 300,   # Fast rise (~300s)
        "tau_decay_flare": 2000, # Slow decay (~2000s)
        "A_flare": 3e-10,        # Flare amplitude (peak flux ~5e-10, continuum ~5e-11)
        "flare_beta": 0.8,       # Flare spectral index (independent, typical ~0.6-1.0)
        "E_iso_wing": 3e51,      # Moderate energy for wing
        "Gamma0_wing": 30,       # Slower than core
        "theta_c_wing": 0.22,    # Wide wing (~13 degrees)
        # Wing microphysics (can differ from core)
        "p_wing": 2.3,           # Slightly different from core
        "eps_e_wing": 0.9,       # Can be different
        "eps_B_wing": 0.005,     # Can be different
        "xi_wing": 0.8,          # Fraction of accelerated electrons
    }
    
    # Convert to sampled space and create Gaussian ball
    pos0 = []
    for p in param_defs:
        if p.name in initial_guess:
            center = initial_guess[p.name]
            if p.scale is Scale.LOG:
                center_log = np.log10(center)
                # Spread of ~0.3 dex (factor of 2)
                pos0.append(np.random.normal(center_log, 0.3, nwalkers))
            else:
                # Spread of ~20% for linear params
                pos0.append(np.random.normal(center, center * 0.2, nwalkers))
        else:
            # Fallback to uniform if not in guess
            if p.scale is Scale.LOG:
                lower_log = np.log10(p.lower)
                upper_log = np.log10(p.upper)
                pos0.append(np.random.uniform(lower_log, upper_log, nwalkers))
            else:
                pos0.append(np.random.uniform(p.lower, p.upper, nwalkers))
    
    pos0 = np.array(pos0).T  # Shape: (nwalkers, ndim)
    
    # Ensure all positions are within bounds
    for i, p in enumerate(param_defs):
        if p.scale is Scale.LOG:
            lower_log = np.log10(p.lower)
            upper_log = np.log10(p.upper)
            pos0[:, i] = np.clip(pos0[:, i], lower_log, upper_log)
        else:
            pos0[:, i] = np.clip(pos0[:, i], p.lower, p.upper)
    
    # Run MCMC
    print("\nRunning MCMC...")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        log_probability,
        args=(param_defs, xrt_data, i_data, args.include_flare, args.include_wing),
    )
    sampler.run_mcmc(pos0, args.nsteps, progress=True)
    
    # Save results
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = "core"
    if args.include_flare:
        phase_name += "_flare"
    if args.include_wing:
        phase_name += "_wing"
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"partial_{phase_name}_{run_ts}"
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
    
    # Save best-fit parameters (detailed format like early_phase)
    model_desc = "Core jet"
    if args.include_flare:
        model_desc += " + Gaussian flare"
    if args.include_wing:
        model_desc += " + Wing jet"
    
    lines = [
        "=== Fit Configuration ===",
        f"Model: {model_desc}",
        f"XRT data: {len(xrt_data)} points, band = {XRT_BAND[0]}-{XRT_BAND[1]} keV",
        f"i-band data: {len(i_data)} points",
        f"XRT time range: {xrt_data['time'].min():.1f} - {xrt_data['time'].max():.1f} s",
        f"i-band time range: {i_data['time'].min():.1f} - {i_data['time'].max():.1f} s",
        f"i-band frequency: {I_BAND_FREQ:.3e} Hz",
        f"Host A_V prior: log10(A_V) ~ Normal({HOST_AV_LOG10_MEAN}, {HOST_AV_LOG10_SIGMA})",
        f"Best log probability: {top_log_probs[0]:.6g}",
        "",
        "=== Best-fit Parameters ===",
        f"{'label':<18} {'sampled':>14} {'physical':>14} {'prior_min':>14} {'prior_max':>14}",
        "-" * 80,
    ]
    
    for label, param_def, sampled in zip(labels, param_defs, top_params[0]):
        if param_def.scale is Scale.LOG:
            physical = 10 ** sampled
        else:
            physical = sampled
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
    
    plot_corner(outdir, labels, max_samples=20000)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")
    
    # Generate best-fit plots
    print("\n" + "=" * 60)
    print("Generating best-fit plots...")
    print("=" * 60)
    
    try:
        from partial_data_plotting import plot_light_curves, plot_spectral_index
        plot_light_curves(outdir, compute_model_components, XRT_BAND, I_BAND_FREQ)
        plot_spectral_index(outdir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
