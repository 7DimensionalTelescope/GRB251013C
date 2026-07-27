#!/usr/bin/env python3
"""
Plotting script for partial_data.py results
Generates best-fit light curves and spectral index comparison plots
"""
from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from VegasAfterglow import Scale

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.io import read_data
from grb.const import D_L, REDSHIFT
from grb.extinction import host_extinction_attenuation
from grb.plotting import set_log_y_limits
from grb.results import latest_result_dir, read_labels
from grb.results import load_best_fit_params as grb_load_best_fit_params
from grb.spectral_index import load_xrt_spectral_index

PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"


def load_best_fit_params(outdir):
    """Load best-fit parameters from saved results

    Wraps grb.results.load_best_fit_params, adding the two extras this module's
    plots need (labels, best_sample) and recognising the legacy "t_peak" flare
    label that predates "t_start_flare".
    """
    outdir = Path(outdir)

    params, include_flare, include_wing = grb_load_best_fit_params(outdir)

    labels = read_labels(outdir / "labels.txt")
    log_probs = np.load(outdir / "log_probs.npy")
    best_sample = np.load(outdir / "samples.npy")[np.argmax(log_probs)]

    # Older runs of this phase named the flare start "t_peak"
    include_flare = include_flare or any("t_peak" in label for label in labels)
    
    return params, include_flare, include_wing, labels, best_sample


def plot_light_curves(outdir, compute_model_components_func, xrt_band, i_band_freq):
    """Generate best-fit light curve plots
    
    Args:
        outdir: Path to results directory
        compute_model_components_func: Function to compute model flux components
        xrt_band: XRT energy band (tuple of min, max keV)
        i_band_freq: i-band frequency in Hz
    """
    outdir = Path(outdir)
    
    # Load data
    xrt_data = read_data("xrt")
    i_data = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    
    # Load best-fit parameters
    params, include_flare, include_wing, labels, best_sample = load_best_fit_params(outdir)
    
    print(f"Plotting light curves for: {outdir.name}")
    print(f"  Model: Core{' + Flare' if include_flare else ''}{' + Wing' if include_wing else ''}")
    
    # Create time grids for plotting (match early_phase: 0.8x to 1.2x data range)
    xrt_t_grid = np.geomspace(xrt_data["time"].min() * 0.8, xrt_data["time"].max() * 1.2, 300)
    i_t_grid = np.geomspace(i_data["time"].min() * 0.8, i_data["time"].max() * 1.2, 300)
    
    # Compute model components
    xrt_components = compute_model_components_func(xrt_t_grid, xrt_band, params, include_flare, include_wing)
    i_components = compute_model_components_func(i_t_grid, i_band_freq, params, include_flare, include_wing)
    
    # Apply extinction and convert i-band to mJy
    if "A_V" in params:
        attenuation = host_extinction_attenuation(i_band_freq, params["A_V"], REDSHIFT)
        i_components_mJy = {key: val * attenuation * 1e26 for key, val in i_components.items()}
    else:
        i_components_mJy = {key: val * 1e26 for key, val in i_components.items()}
    
    # Create figure
    fig, (ax_xrt, ax_i) = plt.subplots(1, 2, figsize=(14, 5))
    
    # XRT panel
    ax_xrt.errorbar(
        xrt_data["time"] / 3600,
        xrt_data["flux"],
        yerr=[
            np.abs(xrt_data["flux_low"].to_numpy(float)),
            xrt_data["flux_high"].to_numpy(float)
        ],
        fmt="o",
        color="black",
        markersize=4,
        alpha=0.7,
        label="XRT data"
    )
    
    # Plot components
    ax_xrt.plot(xrt_t_grid / 3600, xrt_components['total'], "k-", lw=2, label="Total", zorder=10)
    ax_xrt.plot(xrt_t_grid / 3600, xrt_components['core'], "k:", lw=1.5, label="Core FS", alpha=0.8)
    
    if 'reverse' in xrt_components and np.any(xrt_components['reverse'] > 0):
        ax_xrt.plot(xrt_t_grid / 3600, xrt_components['reverse'], "c-.", lw=1.5, label="Reverse shock", alpha=0.8)
    
    if include_flare and np.any(xrt_components['flare'] > 0):
        ax_xrt.plot(xrt_t_grid / 3600, xrt_components['flare'], "r--", lw=1.5, label="Flare", alpha=0.8)
    
    if include_wing and np.any(xrt_components['wing'] > 0):
        ax_xrt.plot(xrt_t_grid / 3600, xrt_components['wing'], "b-.", lw=1.5, label="Wing jet", alpha=0.8)
    
    ax_xrt.set_xlabel("Time since trigger (hr)", fontsize=12)
    ax_xrt.set_ylabel("Flux (0.3-10 keV) [erg/s/cm²]", fontsize=12)
    ax_xrt.set_xscale("log")
    ax_xrt.set_yscale("log")
    ax_xrt.legend(loc="best", fontsize=9)
    ax_xrt.grid(which="both", alpha=0.3)
    ax_xrt.set_title("XRT Light Curve", fontsize=13, fontweight="bold")
    
    # i-band panel
    ax_i.errorbar(
        i_data["time"] / 3600,
        i_data["flux_mJy"],
        yerr=i_data["flux_mJy_error"],
        fmt="o",
        color="red",
        markersize=5,
        alpha=0.7,
        label="i-band data"
    )
    
    # Plot components
    ax_i.plot(i_t_grid / 3600, i_components_mJy['total'], "r-", lw=2, label="Total", zorder=10)
    ax_i.plot(i_t_grid / 3600, i_components_mJy['core'], "r:", lw=1.5, label="Core FS", alpha=0.8)
    
    if 'reverse' in i_components_mJy and np.any(i_components_mJy['reverse'] > 0):
        ax_i.plot(i_t_grid / 3600, i_components_mJy['reverse'], "c-.", lw=1.5, label="Reverse shock", alpha=0.8)
    
    if include_flare and np.any(i_components_mJy['flare'] > 0):
        ax_i.plot(i_t_grid / 3600, i_components_mJy['flare'], "orange", ls="--", lw=1.5, label="Flare", alpha=0.8)
    
    if include_wing and np.any(i_components_mJy['wing'] > 0):
        ax_i.plot(i_t_grid / 3600, i_components_mJy['wing'], "b-.", lw=1.5, label="Wing jet", alpha=0.8)
    
    ax_i.set_xlabel("Time since trigger (hr)", fontsize=12)
    ax_i.set_ylabel("Flux density (i-band) [mJy]", fontsize=12)
    ax_i.set_xscale("log")
    ax_i.set_yscale("log")
    ax_i.legend(loc="best", fontsize=9)
    ax_i.grid(which="both", alpha=0.3)
    ax_i.set_title("i-band Light Curve", fontsize=13, fontweight="bold")
    
    # Set y-axis limits based on data only (not model)
    xrt_err_low = np.abs(xrt_data["flux_low"].to_numpy(float))
    xrt_err_high = xrt_data["flux_high"].to_numpy(float)
    xrt_flux = xrt_data["flux"].to_numpy(float)
    
    set_log_y_limits(
        ax_xrt,
        xrt_flux - xrt_err_low,
        xrt_flux + xrt_err_high,
    )
    
    i_flux = i_data["flux_mJy"].to_numpy(float)
    i_err = i_data["flux_mJy_error"].to_numpy(float)
    
    set_log_y_limits(
        ax_i,
        i_flux - i_err,
        i_flux + i_err,
    )
    
    plt.tight_layout()
    
    # Save
    outfile = outdir / "bestfit_lc.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"  ✓ Saved: {outfile}")
    plt.close()


def plot_spectral_index(outdir):
    """Generate spectral index comparison plot
    
    Args:
        outdir: Path to results directory
    """
    outdir = Path(outdir)
    
    # Load best-fit parameters
    params, include_flare, include_wing, labels, best_sample = load_best_fit_params(outdir)
    
    print(f"Plotting spectral index comparison...")
    
    # Load XRT spectral index data
    try:
        beta_xrt_data = load_xrt_spectral_index()
    except Exception as e:
        print(f"  ⚠ Could not load XRT spectral index: {e}")
        print(f"  Skipping spectral index plot")
        return
    
    # Model spectral index from microphysics
    p = params["p"]
    beta_model_slow = (p - 1) / 2  # Slow cooling
    beta_model_fast = p / 2        # Fast cooling
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot XRT data (with asymmetric errors)
    ax.errorbar(
        beta_xrt_data["time"] / 3600,
        beta_xrt_data["beta"],
        yerr=[beta_xrt_data["beta_err_low"], beta_xrt_data["beta_err_high"]],
        fmt="o",
        color="C0",
        markersize=6,
        capsize=3,
        alpha=0.7,
        label="XRT spectral index"
    )
    
    # Plot model predictions
    ax.axhline(beta_model_slow, color="red", ls="--", lw=2, 
               label=f"Slow cooling: β = (p-1)/2 = {beta_model_slow:.3f}")
    ax.axhline(beta_model_fast, color="orange", ls=":", lw=2,
               label=f"Fast cooling: β = p/2 = {beta_model_fast:.3f}")
    
    ax.set_xlabel("Time since trigger (hr)", fontsize=12)
    ax.set_ylabel("Spectral index β (F_ν ∝ ν^-β)", fontsize=12)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"XRT Spectral Index (p = {p:.3f})", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    
    # Save
    outfile = outdir / "spectral_index_comparison.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"  ✓ Saved: {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot partial_data.py fit results")
    parser.add_argument("--outdir", default=None, help="Output directory (default: latest)")
    args = parser.parse_args()
    
    # Import here to avoid circular dependency
    from partial_data import make_param_defs, compute_model_components, XRT_BAND, I_BAND_FREQ
    
    # Find output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = latest_result_dir(FIT_RESULTS_DIR, "partial_")
    
    if not outdir.exists():
        print(f"Error: {outdir} does not exist")
        return 1
    
    print(f"\n{'=' * 60}")
    print(f"Plotting partial_data.py results")
    print(f"{'=' * 60}\n")
    print(f"Results directory: {outdir}\n")
    
    # Generate plots
    plot_light_curves(outdir, compute_model_components, XRT_BAND, I_BAND_FREQ)
    plot_spectral_index(outdir)
    
    print(f"\n{'=' * 60}")
    print("✓ All plots generated successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
