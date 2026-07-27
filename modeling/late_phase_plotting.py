from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from VegasAfterglow.units import mJy

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.io import read_data
from grb.const import REDSHIFT
from utils import (
    XRT_EXCLUDE_TIME_RANGE,
    compute_break_frequencies,
    host_extinction_attenuation,
    latest_result_dir,
    load_xrt_spectral_index,
    plot_corner,
    read_labels,
    set_log_y_limits,
)
from spectral_index_interpolator import get_spectral_index_calculator


C_AA_PER_S = 2.99792458e18
XRT_LABEL = "XRT"
PROJECT_DIR = Path(__file__).absolute().parent
FIT_RESULTS_DIR = PROJECT_DIR / "fit_results"


def plot_best_fit(
    outdir,
    theta,
    param_defs,
    early_params,
    fixed_model,
    xrt_data,
    optical_data,
    make_core_model,
    to_physical,
    xrt_band,
    xrt_flux_error,
):
    outdir = Path(outdir)
    params = to_physical(theta, param_defs)
    fitted_core_model = make_core_model(params, early_params)

    t_min = min([xrt_data["time"].min()] + [dataset.time_s.min() for dataset in optical_data])
    t_max = max([xrt_data["time"].max()] + [dataset.time_s.max() for dataset in optical_data])
    t_grid = np.geomspace(t_min * 0.8, t_max * 1.2, 300)
    lc_data = [dataset for dataset in optical_data if not dataset.name.startswith("7DT ")]
    sdt_data = [dataset for dataset in optical_data if dataset.name.startswith("7DT ")]

    xrt_fixed = np.asarray(fixed_model.flux(t_grid, xrt_band[0], xrt_band[1], 10).total)
    xrt_new_core = np.asarray(fitted_core_model.flux(t_grid, xrt_band[0], xrt_band[1], 10).total)

    fig = plt.figure(figsize=(16, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9])
    ax_xrt = fig.add_subplot(gs[0])
    ax_opt = fig.add_subplot(gs[1])
    ax_sed = fig.add_subplot(gs[2])

    xrt_err = xrt_flux_error(xrt_data)
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
            yerr=xrt_flux_error(flare_xrt_data),
            fmt=".",
            color="gray",
            alpha=0.6,
            label="XRT flare data",
        )
    ax_xrt.errorbar(
        xrt_data["time"] / 3600,
        xrt_data["flux"],
        yerr=xrt_err,
        fmt=".",
        color="black",
        label="XRT data",
    )
    ax_xrt.plot(t_grid / 3600, xrt_fixed + xrt_new_core, color="black", label="XRT total")
    ax_xrt.plot(t_grid / 3600, xrt_fixed, color="black", ls=":", label="early tophat")
    ax_xrt.plot(t_grid / 3600, xrt_new_core, color="black", ls="-.", label="core tophat")
    ax_xrt.set_xlabel("Time since trigger [hr]")
    ax_xrt.set_ylabel(r"Flux [erg cm$^{-2}$ s$^{-1}$]")
    ax_xrt.legend(fontsize=8)

    colors = {
        "i": "red",
        "Leavitt Rc": "darkorange",
        "Leavitt Ic": "darkred",
    }
    for dataset in lc_data:
        color = colors.get(dataset.name, "slategray")
        ax_opt.errorbar(
            dataset.time_s / 3600,
            dataset.flux_mjy,
            yerr=dataset.flux_err_mjy,
            fmt=".",
            color=color,
            alpha=0.7,
            label=dataset.name,
        )

    model_bands = [
        dataset for dataset in lc_data
        if dataset.name in {"i", "Leavitt Rc", "Leavitt Ic"}
    ]
    seen = set()
    for dataset in model_bands:
        if dataset.name in seen:
            continue
        seen.add(dataset.name)
        nu = dataset.frequency_hz
        fixed = np.asarray(fixed_model.flux_density(t_grid, nu * np.ones_like(t_grid)).total)
        new_core = np.asarray(fitted_core_model.flux_density(t_grid, nu * np.ones_like(t_grid)).total)
        attenuation = host_extinction_attenuation(nu * np.ones_like(t_grid), params["A_V"], REDSHIFT)
        color = colors[dataset.name]
        total = (fixed + new_core) * attenuation / mJy
        ax_opt.plot(t_grid / 3600, total, color=color, label=f"{dataset.name} total")
        ax_opt.plot(t_grid / 3600, fixed * attenuation / mJy, color=color, ls=":", alpha=0.7)
        ax_opt.plot(t_grid / 3600, new_core * attenuation / mJy, color=color, ls="-.", alpha=0.7)

    ax_opt.set_xlabel("Time since trigger [hr]")
    ax_opt.set_ylabel("Flux density [mJy]")
    handles, labels = ax_opt.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_opt.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)

    if sdt_data:
        sdt_time = np.array([dataset.time_s[0] for dataset in sdt_data])
        sdt_nu = np.array([dataset.frequency_hz for dataset in sdt_data])
        sdt_wavelength = C_AA_PER_S / sdt_nu
        sdt_flux = np.array([dataset.flux_mjy[0] for dataset in sdt_data])
        sdt_flux_err = np.array([dataset.flux_err_mjy[0] for dataset in sdt_data])
        sed_time = float(np.median(sdt_time))
        wavelength_grid = np.geomspace(sdt_wavelength.min() * 0.9, sdt_wavelength.max() * 1.1, 300)
        nu_grid = C_AA_PER_S / wavelength_grid

        fixed_sed = np.asarray(fixed_model.flux_density(sed_time * np.ones_like(nu_grid), nu_grid).total)
        new_core_sed = np.asarray(fitted_core_model.flux_density(sed_time * np.ones_like(nu_grid), nu_grid).total)
        attenuation = host_extinction_attenuation(nu_grid, params["A_V"], REDSHIFT)
        total_sed = (fixed_sed + new_core_sed) * attenuation / mJy

        ax_sed.errorbar(
            sdt_wavelength,
            sdt_flux,
            yerr=sdt_flux_err,
            fmt=".",
            color="slategray",
            alpha=0.8,
            label="7DT data",
        )
        ax_sed.plot(
            wavelength_grid,
            total_sed,
            color="black",
            label=f"total at {sed_time / 3600:.2f} hr",
        )
        ax_sed.plot(wavelength_grid, fixed_sed * attenuation / mJy, color="black", ls=":", label="early tophat")
        ax_sed.plot(wavelength_grid, new_core_sed * attenuation / mJy, color="black", ls="-.", label="core tophat")
        ax_sed.set_xlabel(r"Wavelength [$\AA$]")
        ax_sed.set_ylabel("Flux density [mJy]")
        ax_sed.legend(fontsize=8)

    for ax in (ax_xrt, ax_opt):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both", alpha=0.25)
    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")
    ax_sed.grid(which="both", alpha=0.25)
    set_log_y_limits(
        ax_xrt,
        xrt_data["flux"].to_numpy(float) - xrt_err,
        xrt_data["flux"].to_numpy(float) + xrt_err,
        flare_xrt_data["flux"].to_numpy(float) - xrt_flux_error(flare_xrt_data),
        flare_xrt_data["flux"].to_numpy(float) + xrt_flux_error(flare_xrt_data),
        xrt_fixed + xrt_new_core,
    )
    set_log_y_limits(
        ax_opt,
        *[
            value
            for dataset in lc_data
            for value in (
                dataset.flux_mjy - dataset.flux_err_mjy,
                dataset.flux_mjy + dataset.flux_err_mjy,
            )
        ],
    )
    if sdt_data:
        set_log_y_limits(
            ax_sed,
            sdt_flux - sdt_flux_err,
            sdt_flux + sdt_flux_err,
            total_sed,
        )

    fig.tight_layout()
    fig.savefig(outdir / "bestfit_lc.png", dpi=200)
    plt.close(fig)


def plot_spectral_index_comparison(best_params, param_labels, outdir, xrt_index_data=None):
    """
    Generate spectral index comparison plot showing observed vs fitted photon index.
    
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
            # Filter to late phase only (after flare)
            from utils import XRT_FLARE_START_TIME, XRT_FLARE_END_TIME
            mask = xrt_index_data["time"] > XRT_FLARE_END_TIME
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


def regenerate_plot(result_dir=None, early_dir=None, corner_samples=20000, skip_corner=False):
    from late_phase import (
        XRT_BAND,
        early_dir_from_late_result,
        ensure_saved_labels_match,
        find_latest_early_dir,
        load_early_core_params,
        load_late_phase_data,
        make_fixed_early_wing_model,
        make_core_model,
        make_param_defs,
        make_wing_model,
        sampled_labels,
        save_bestfit_params,
        to_physical,
        xrt_flux_error,
    )

    result_dir = Path(result_dir) if result_dir else latest_result_dir(FIT_RESULTS_DIR, "late_phase_")
    if early_dir is not None:
        early_dir = Path(early_dir)
    else:
        early_dir = early_dir_from_late_result(result_dir) or find_latest_early_dir()

    early_params = load_early_core_params(early_dir)
    free_w = any(label.endswith("_w") for label in read_labels(result_dir / "labels.txt"))
    xrt_data, optical_data, _ = load_late_phase_data(include_spectral_index=False)
    param_defs = make_param_defs(early_params, free_w)
    labels = sampled_labels(param_defs)
    ensure_saved_labels_match(result_dir, labels)
    theta = np.load(result_dir / "top_k_params.npy")[0]
    log_prob = np.load(result_dir / "top_k_log_probs.npy")[0]
    wing_model = make_wing_model(to_physical(theta, param_defs), early_params) if free_w else make_fixed_early_wing_model(early_params)

    plot_best_fit(
        result_dir,
        theta,
        param_defs,
        early_params,
        wing_model,
        xrt_data,
        optical_data,
        make_core_model,
        to_physical,
        XRT_BAND,
        xrt_flux_error,
    )
    save_bestfit_params(result_dir, theta, log_prob, param_defs, early_params, early_dir, xrt_data, optical_data, free_w)
    
    # Generate spectral index comparison plot if data is available
    try:
        xrt_index_data = load_xrt_spectral_index()
        from utils import XRT_FLARE_END_TIME
        mask = xrt_index_data["time"] > XRT_FLARE_END_TIME
        xrt_index_data = {k: v[mask] for k, v in xrt_index_data.items()}
        if len(xrt_index_data.get("time", [])) > 0:
            plot_spectral_index_comparison(theta, labels, result_dir, xrt_index_data)
    except Exception as e:
        print(f"Warning: Could not generate spectral index comparison plot: {e}")
    
    if not skip_corner:
        plot_corner(result_dir, labels, max_samples=corner_samples)

    print(f"Result directory: {result_dir}")
    print(f"Early _w component directory: {early_dir}")
    print(f"Best-fit plot saved to: {result_dir / 'bestfit_lc.png'}")
    if not skip_corner:
        print(f"Corner plot saved to: {result_dir / 'corner_plot.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate the latest late-phase fit plots.")
    parser.add_argument("--result-dir", default=None, help="Specific late_phase result directory to replot.")
    parser.add_argument("--early-dir", default=None, help="Override the fixed early _w component result directory.")
    parser.add_argument("--corner-samples", type=int, default=20000)
    parser.add_argument("--skip-corner", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    regenerate_plot(
        result_dir=args.result_dir,
        early_dir=args.early_dir,
        corner_samples=args.corner_samples,
        skip_corner=args.skip_corner,
    )


if __name__ == "__main__":
    main()

