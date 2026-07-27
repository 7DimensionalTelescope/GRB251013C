from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from VegasAfterglow.units import Hz, mJy, sec

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.io import read_data
from grb.const import REDSHIFT, XRT_EXCLUDE_TIME_RANGE
from grb.plotting import plot_corner, set_log_y_limits
from grb.results import latest_result_dir
from grb.spectral_index import compute_break_frequencies, load_xrt_spectral_index
from grb.utils import flux_error, model_array
from spectral_index_interpolator import get_spectral_index_calculator


PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"


def plot_best_fit(fitter, best_params, xrt_data, i_data, outdir, xrt_band, host_av_param=None):
    outdir = Path(outdir)
    xrt_t_grid = np.geomspace(xrt_data["time"].min() * 0.8, xrt_data["time"].max() * 1.2, 200)
    i_t_grid = np.geomspace(i_data["time"].min() * 0.8, i_data["time"].max() * 1.2, 200)

    xrt_output = fitter.flux(best_params, xrt_t_grid * sec, xrt_band, num_points=10)
    i_output = fitter.flux_density_grid(
        best_params,
        i_t_grid * sec,
        np.array([float(i_data["frequency_Hz"].iloc[0])]) * Hz,
    )
    xrt_model = model_array(xrt_output.total)
    xrt_fwd = model_array(xrt_output.fwd)
    xrt_rvs = model_array(xrt_output.rvs)
    i_model = model_array(i_output.total)
    i_fwd = model_array(i_output.fwd)
    i_rvs = model_array(i_output.rvs)

    fig, (ax_xrt, ax_i) = plt.subplots(1, 2, figsize=(12, 4.5))

    xrt_err = flux_error(xrt_data)
    all_xrt_data = read_data("xrt")
    flare_xrt_data = all_xrt_data[all_xrt_data["time"].between(*XRT_EXCLUDE_TIME_RANGE)]
    ax_xrt.axvspan(
        XRT_EXCLUDE_TIME_RANGE[0] / 3600,
        XRT_EXCLUDE_TIME_RANGE[1] / 3600,
        color="gray",
        alpha=0.15,
        label="flare excluded from fit",
    )
    if len(flare_xrt_data) > 0:
        ax_xrt.errorbar(
            flare_xrt_data["time"] / 3600,
            flare_xrt_data["flux"],
            yerr=flux_error(flare_xrt_data),
            fmt="o",
            color="gray",
            alpha=0.6,
            label="XRT flare data",
        )
    ax_xrt.errorbar(
        xrt_data["time"] / 3600,
        xrt_data["flux"],
        yerr=xrt_err,
        fmt="o",
        color="black",
        label="XRT data",
    )
    ax_xrt.plot(xrt_t_grid / 3600, xrt_model, color="black", label="XRT total")
    ax_xrt.plot(xrt_t_grid / 3600, xrt_fwd, color="black", ls=":", label="XRT FS")
    ax_xrt.plot(xrt_t_grid / 3600, xrt_rvs, color="black", ls="-.", label="XRT RS")
    ax_xrt.set_xlabel("Time since trigger [hr]")
    ax_xrt.set_ylabel(r"Flux [erg cm$^{-2}$ s$^{-1}$]")
    ax_xrt.legend()

    ax_i.errorbar(
        i_data["time"] / 3600,
        i_data["flux_mJy"],
        yerr=i_data["flux_mJy_error"],
        fmt="o",
        color="red",
        label="i data",
    )
    ax_i.plot(i_t_grid / 3600, i_model / mJy, color="red", label="i total")
    ax_i.plot(i_t_grid / 3600, i_fwd / mJy, color="red", ls=":", label="i FS")
    ax_i.plot(i_t_grid / 3600, i_rvs / mJy, color="red", ls="-.", label="i RS")
    ax_i.set_xlabel("Time since trigger [hr]")
    ax_i.set_ylabel("Flux density [mJy]")
    ax_i.legend()

    for ax in (ax_xrt, ax_i):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both", alpha=0.25)

    set_log_y_limits(
        ax_xrt,
        xrt_data["flux"].to_numpy(float) - xrt_err,
        xrt_data["flux"].to_numpy(float) + xrt_err,
        flare_xrt_data["flux"].to_numpy(float) - flux_error(flare_xrt_data),
        flare_xrt_data["flux"].to_numpy(float) + flux_error(flare_xrt_data),
        xrt_model,
    )
    set_log_y_limits(
        ax_i,
        i_data["flux_mJy"].to_numpy(float) - i_data["flux_mJy_error"].to_numpy(float),
        i_data["flux_mJy"].to_numpy(float) + i_data["flux_mJy_error"].to_numpy(float),
        i_model / mJy,
    )

    fig.tight_layout()
    fig.savefig(outdir / "bestfit_lc.png", dpi=200)
    plt.close(fig)


def plot_spectral_index_comparison(best_params, param_labels, outdir, xrt_index_data=None):
    """
    Plot comparison between observed XRT photon index and model prediction
    using smooth spectral index transitions (Granot & Sari 2001).
    
    Args:
        best_params: array of best-fit parameters (in sampled space)
        param_labels: list of parameter labels
        outdir: output directory path
        xrt_index_data: dict with XRT spectral index measurements
    """
    outdir = Path(outdir)
    
    # Load XRT spectral index data if not provided
    if xrt_index_data is None:
        try:
            xrt_index_data = load_xrt_spectral_index()
            # Filter to early phase only (before flare)
            from grb.const import XRT_FLARE_START_TIME
            mask = xrt_index_data["time"] < XRT_FLARE_START_TIME
            xrt_index_data = {k: v[mask] for k, v in xrt_index_data.items()}
        except Exception as e:
            print(f"Warning: Could not load XRT spectral index data: {e}")
            return
    
    if len(xrt_index_data["time"]) == 0:
        print("Warning: No XRT spectral index data available for plotting")
        return
    
    # Convert best_params to physical parameter dict
    params_dict = {}
    for label, value in zip(param_labels, best_params):
        if label.startswith("log10_"):
            param_name = label.replace("log10_", "")
            params_dict[param_name] = 10 ** value
        else:
            params_dict[label] = value
    
    # Get spectral index calculator
    calc = get_spectral_index_calculator()
    
    # XRT center frequency (~1.2 keV)
    nu_xrt = 3e17  # Hz
    
    # Time grid for smooth model curve
    t_grid = np.geomspace(
        xrt_index_data["time"].min() * 0.5,
        xrt_index_data["time"].max() * 2.0,
        200
    )
    
    # Compute model spectral index using smooth transitions
    beta_model_smooth = np.zeros_like(t_grid)
    beta_model_step_slow = np.zeros_like(t_grid)
    beta_model_step_fast = np.zeros_like(t_grid)
    
    for i, t in enumerate(t_grid):
        # Compute break frequencies at this time
        breaks = compute_break_frequencies(params_dict, REDSHIFT, t)
        
        # Smooth spectral index
        beta_model_smooth[i] = calc.beta_at_frequency(
            nu_xrt, breaks["nu_m"], breaks["nu_c"], params_dict["p"]
        )
        
        # Step function approximations for comparison (G&S Table 2)
        # Slow cooling regime: nu_m < nu_XRT < nu_c
        if nu_xrt < breaks["nu_c"]:
            # Below nu_c: beta = (1-p)/2
            beta_model_step_slow[i] = (1.0 - params_dict["p"]) / 2.0
        else:
            # Above nu_c: beta = -p/2
            beta_model_step_slow[i] = -params_dict["p"] / 2.0
        
        # Fast cooling: beta = -p/2 everywhere above nu_c
        beta_model_step_fast[i] = -params_dict["p"] / 2.0
    
    # Convert beta (G&S) to photon index Gamma = 1 - beta
    gamma_model_smooth = 1.0 - beta_model_smooth
    gamma_model_step_slow = 1.0 - beta_model_step_slow
    gamma_model_step_fast = 1.0 - beta_model_step_fast
    
    # Observed photon index (Gamma from XRT, stored as beta in G&S convention)
    gamma_obs = 1.0 - xrt_index_data["beta"]
    gamma_obs_err_low = xrt_index_data["beta_err_low"]
    gamma_obs_err_high = xrt_index_data["beta_err_high"]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Upper panel: Photon index vs time
    ax1.errorbar(
        xrt_index_data["time"] / 3600,
        gamma_obs,
        yerr=[gamma_obs_err_low, gamma_obs_err_high],
        fmt='o',
        color='black',
        markersize=6,
        capsize=3,
        label='XRT observed Γ',
        zorder=10
    )
    
    ax1.plot(
        t_grid / 3600,
        gamma_model_smooth,
        color='blue',
        linewidth=2.5,
        label='Model (smooth transitions)',
        zorder=5
    )
    
    ax1.plot(
        t_grid / 3600,
        gamma_model_step_slow,
        color='red',
        linewidth=1.5,
        linestyle='--',
        label='Step function (slow cooling)',
        alpha=0.7,
        zorder=3
    )
    
    ax1.set_ylabel('Photon Index Γ', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xscale('log')
    ax1.set_title(f'XRT Spectral Index Evolution (p = {params_dict["p"]:.3f})',
                  fontsize=13, fontweight='bold')
    
    # Lower panel: Residuals
    # Interpolate model to observation times
    gamma_model_at_obs = np.interp(
        xrt_index_data["time"],
        t_grid,
        gamma_model_smooth
    )
    gamma_step_at_obs = np.interp(
        xrt_index_data["time"],
        t_grid,
        gamma_model_step_slow
    )
    
    residuals_smooth = gamma_obs - gamma_model_at_obs
    residuals_step = gamma_obs - gamma_step_at_obs
    
    ax2.errorbar(
        xrt_index_data["time"] / 3600,
        residuals_smooth,
        yerr=[gamma_obs_err_low, gamma_obs_err_high],
        fmt='o',
        color='blue',
        markersize=5,
        capsize=3,
        label='Smooth model residuals',
        alpha=0.8
    )
    
    ax2.errorbar(
        xrt_index_data["time"] / 3600,
        residuals_step,
        yerr=[gamma_obs_err_low, gamma_obs_err_high],
        fmt='s',
        color='red',
        markersize=4,
        capsize=3,
        label='Step function residuals',
        alpha=0.6
    )
    
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time since trigger [hr]', fontsize=12)
    ax2.set_ylabel('Residual Γ', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xscale('log')
    
    # Calculate and display chi-squared
    chi2_smooth = np.sum((residuals_smooth / np.maximum(gamma_obs_err_low, gamma_obs_err_high)) ** 2)
    chi2_step = np.sum((residuals_step / np.maximum(gamma_obs_err_low, gamma_obs_err_high)) ** 2)
    reduced_chi2_smooth = chi2_smooth / len(residuals_smooth)
    reduced_chi2_step = chi2_step / len(residuals_step)
    
    textstr = (f'Smooth: χ²/dof = {chi2_smooth:.1f}/{len(residuals_smooth)} = {reduced_chi2_smooth:.2f}\n'
               f'Step:   χ²/dof = {chi2_step:.1f}/{len(residuals_step)} = {reduced_chi2_step:.2f}')
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    fig.savefig(outdir / "spectral_index_comparison.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Spectral index comparison plot saved to: {outdir / 'spectral_index_comparison.png'}")
    print(f"  Smooth model: χ²/dof = {reduced_chi2_smooth:.2f}")
    print(f"  Step function: χ²/dof = {reduced_chi2_step:.2f}")


def regenerate_plot(result_dir=None, corner_samples=20000, skip_corner=False):
    try:
        from VegasAfterglow.fitting.utils import _build_transformer
    except ImportError:
        from VegasAfterglow.runner import _build_transformer

    from early_phase import (
        HOST_AV_PARAM,
        XRT_BAND,
        load_fit_data,
        make_fitter,
        make_param_defs,
        param_defs_for_saved_result,
        save_bestfit_params,
    )

    result_dir = Path(result_dir) if result_dir else latest_result_dir(FIT_RESULTS_DIR, "early_phase_")
    xrt_data, i_data, xrt_index_data = load_fit_data(include_spectral_index=True)
    fitter = make_fitter(xrt_data, i_data)
    param_defs = param_defs_for_saved_result(result_dir, make_param_defs())
    fitter._to_params = _build_transformer(param_defs)
    best_params = np.load(result_dir / "top_k_params.npy")[0]
    labels = [line.strip() for line in (result_dir / "labels.txt").read_text().splitlines() if line.strip()]

    plot_best_fit(fitter, best_params, xrt_data, i_data, result_dir, XRT_BAND, HOST_AV_PARAM)
    save_bestfit_params(result_dir, param_defs, xrt_data, i_data)
    
    # Generate spectral index comparison plot if data is available
    if xrt_index_data is not None and len(xrt_index_data.get("time", [])) > 0:
        plot_spectral_index_comparison(best_params, labels, result_dir, xrt_index_data)
    
    if not skip_corner:
        plot_corner(result_dir, max_samples=corner_samples)

    print(f"Result directory: {result_dir}")
    print(f"Best-fit plot saved to: {result_dir / 'bestfit_lc.png'}")
    if not skip_corner:
        print(f"Corner plot saved to: {result_dir / 'corner_plot.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate the latest early-phase fit plots.")
    parser.add_argument("--result-dir", default=None, help="Specific early_phase result directory to replot.")
    parser.add_argument("--corner-samples", type=int, default=20000)
    parser.add_argument("--skip-corner", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    regenerate_plot(
        result_dir=args.result_dir,
        corner_samples=args.corner_samples,
        skip_corner=args.skip_corner,
    )


if __name__ == "__main__":
    main()

