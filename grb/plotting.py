"""Figure generation for afterglow fit results.

Light curves with component breakdown, the 7DT spectrum panel, the XRT
spectral-index comparison, and the posterior corner plot.
"""
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from .const import REDSHIFT, XRT_BAND, XRT_NU_HI, XRT_NU_LO
from .extinction import host_extinction_attenuation
from .functions import norris_flare
from .likelihood import spectral_index_model
from .modeling import load_all_optical_data, make_core_model, make_wing_model
from .results import load_best_fit_params, read_labels
from .spectral_index import compute_break_frequencies, load_xrt_spectral_index
from .utils import model_array

C_AA_PER_S = 2.99792458e18  # Speed of light in Angstrom/s


def plot_corner(outdir, labels=None, max_samples=20000, seed=42):
    outdir = Path(outdir)
    samples = np.load(outdir / "samples.npy")
    samples = samples.reshape(-1, samples.shape[-1])
    log_probs = np.load(outdir / "log_probs.npy").reshape(-1)
    if labels is None:
        labels = read_labels(outdir / "labels.txt")

    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(log_probs)
    samples = samples[finite]
    log_probs = log_probs[finite]

    # Best-fit (max log-prob) sample, marked with red crosshairs
    best_fit = samples[np.argmax(log_probs)]

    if len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        samples = samples[rng.choice(len(samples), max_samples, replace=False)]

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        bins=30,
        smooth=True,
        truths=best_fit,
        truth_color="red",
    )
    fig.savefig(outdir / "corner_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def set_log_y_limits(ax, *values, lower_factor=0.5, upper_factor=2.0):
    positive = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array) & (array > 0)]
        if len(array):
            positive.append(array)
    if not positive:
        return

    combined = np.concatenate(positive)
    ax.set_ylim(combined.min() * lower_factor, combined.max() * upper_factor)


def compute_model_components(params, times, frequency, xrt_band, include_flare, include_wing):
    """Compute individual model components

    Returns dict with keys: 'core_fs', 'core_rs', 'flare', 'wing', 'total'
    """
    # Core model (FS + RS)
    core_model = make_core_model(params)

    # Check if XRT or optical
    is_xrt = (xrt_band is not None)

    if is_xrt:
        # XRT flux
        core_output = core_model.flux(times, xrt_band[0], xrt_band[1], 10)
        core_fs = model_array(core_output.fwd).copy()
        core_rs = model_array(core_output.rvs).copy()

        # Wing
        wing_flux = np.zeros_like(core_fs)
        if include_wing and "E_iso_wing" in params:
            wing_model = make_wing_model(params)
            wing_output = wing_model.flux(times, xrt_band[0], xrt_band[1], 10)
            wing_flux = model_array(wing_output.total).copy()

        # Flare
        flare_flux = np.zeros_like(core_fs)
        if include_flare and "t_start_flare" in params:
            flare_flux = norris_flare(times, params["t_start_flare"],
                                     params["tau_rise_flare"], params["tau_decay_flare"],
                                     params["A_flare"])
    else:
        # Optical flux density
        nu_array = frequency * np.ones_like(times)
        core_output = core_model.flux_density(times, nu_array)
        core_fs = model_array(core_output.fwd).copy()
        core_rs = model_array(core_output.rvs).copy()

        # Wing
        wing_flux = np.zeros_like(core_fs)
        if include_wing and "E_iso_wing" in params:
            wing_model = make_wing_model(params)
            wing_output = wing_model.flux_density(times, nu_array)
            wing_flux = model_array(wing_output.total).copy()

        # Flare (with spectral scaling)
        flare_flux = np.zeros_like(core_fs)
        if include_flare and "t_start_flare" in params:
            nu_xrt_min = XRT_NU_LO
            nu_xrt_max = XRT_NU_HI
            beta_flare = params.get("flare_beta", 0.8)

            flare_temporal = norris_flare(times, params["t_start_flare"],
                                         params["tau_rise_flare"], params["tau_decay_flare"],
                                         params["A_flare"])

            if abs(beta_flare - 1.0) < 0.01:
                K = flare_temporal / np.log(nu_xrt_max / nu_xrt_min)
            else:
                K = flare_temporal * (1 - beta_flare) / (nu_xrt_max**(1-beta_flare) - nu_xrt_min**(1-beta_flare))

            flare_flux = K * frequency**(-beta_flare)

        # Apply host extinction
        if "A_V" in params:
            attenuation = host_extinction_attenuation(nu_array, params["A_V"], REDSHIFT)
            core_fs = core_fs * attenuation
            core_rs = core_rs * attenuation
            wing_flux = wing_flux * attenuation
            flare_flux = flare_flux * attenuation

        # Convert to mJy
        core_fs = core_fs * 1e26
        core_rs = core_rs * 1e26
        wing_flux = wing_flux * 1e26
        flare_flux = flare_flux * 1e26

    total = core_fs + core_rs + wing_flux + flare_flux

    return {
        'core_fs': core_fs,
        'core_rs': core_rs,
        'flare': flare_flux,
        'wing': wing_flux,
        'total': total,
    }


def plot_light_curves(outdir):
    """Generate best-fit light curve plots with component contributions and 7DT spectrum"""
    outdir = Path(outdir)

    # Load best-fit parameters
    params, include_flare, include_wing = load_best_fit_params(outdir)

    # Load data
    xrt_data, optical_datasets = load_all_optical_data()

    # Separate optical LC data from 7DT spectrum data
    lc_data = [d for d in optical_datasets if not d['name'].startswith('7DT_')]
    sdt_data = [d for d in optical_datasets if d['name'].startswith('7DT_')]

    # Create time grid for model
    t_min = min(xrt_data['time'].min(), min(d['time'].min() for d in lc_data))
    t_max = max(xrt_data['time'].max(), max(d['time'].max() for d in lc_data))
    t_grid = np.geomspace(t_min * 0.8, t_max * 1.2, 300)

    # Compute XRT model components
    xrt_components = compute_model_components(params, t_grid, None, XRT_BAND,
                                              include_flare, include_wing)

    # Create 3-panel figure
    fig = plt.figure(figsize=(16, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9])
    ax_xrt = fig.add_subplot(gs[0])
    ax_opt = fig.add_subplot(gs[1])
    ax_sed = fig.add_subplot(gs[2])

    # === XRT Panel ===
    xrt_times_hr = xrt_data['time'] / 3600
    ax_xrt.errorbar(xrt_times_hr, xrt_data['flux'], yerr=xrt_data['flux_error'],
                   fmt='.', color='black', alpha=0.7, markersize=6, label='XRT data')

    # Plot components
    t_grid_hr = t_grid / 3600
    ax_xrt.plot(t_grid_hr, xrt_components['total'], 'k-', lw=2, label='Total')
    ax_xrt.plot(t_grid_hr, xrt_components['core_fs'], 'k:', lw=1.5, label='Core FS')
    ax_xrt.plot(t_grid_hr, xrt_components['core_rs'], 'c-.', lw=1.5, label='Reverse Shock')
    if include_wing:
        ax_xrt.plot(t_grid_hr, xrt_components['wing'], 'b--', lw=1.5, label='Wing')
    if include_flare and np.any(xrt_components['flare'] > 0):
        ax_xrt.plot(t_grid_hr, xrt_components['flare'], color='red', ls=(0, (5, 1)), lw=1.5, alpha=0.7, label='Flare')

    ax_xrt.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_xrt.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=11)
    ax_xrt.set_xscale('log')
    ax_xrt.set_yscale('log')
    ax_xrt.legend(fontsize=8)
    ax_xrt.grid(which='both', alpha=0.25)
    ax_xrt.set_title('XRT 0.3-10 keV', fontsize=12, fontweight='bold')

    # === Optical Panel (i-band + Leavitt only) ===
    colors = {
        'i-band': 'red',
        'Leavitt_Rc': 'darkorange',
        'Leavitt_Ic': 'darkred',
    }

    # Offset factors for clarity (avoid overplotting)
    offset_factors = {
        'i-band': 1.0,
        'Leavitt_Rc': 1.5,  # Multiply by 1.5 for visibility
        'Leavitt_Ic': 2.0,  # Multiply by 2.0 for visibility
    }

    for dataset in lc_data:
        color = colors.get(dataset['name'], 'slategray')
        offset = offset_factors.get(dataset['name'], 1.0)
        times_hr = dataset['time'] / 3600

        # Apply offset factor for Leavitt data
        flux_plot = dataset['flux_mJy'] * offset
        flux_err_plot = dataset['flux_err'] * offset

        label = dataset['name']
        if offset != 1.0:
            label += f' (×{offset:.1f})'

        ax_opt.errorbar(times_hr, flux_plot, yerr=flux_err_plot,
                       fmt='.', color=color, alpha=0.7, markersize=6, label=label)

    # Plot model components for main bands (with same offset factors)
    for dataset in lc_data:
        if dataset['name'] not in colors:
            continue

        color = colors[dataset['name']]
        offset = offset_factors.get(dataset['name'], 1.0)
        nu = dataset['frequency']

        # Compute components
        opt_components = compute_model_components(params, t_grid, nu, None,
                                                  include_flare, include_wing)

        # Apply same offset factor to model
        ax_opt.plot(t_grid_hr, opt_components['total'] * offset, color=color, lw=2)
        ax_opt.plot(t_grid_hr, opt_components['core_fs'] * offset, color=color, ls=':', lw=1.5, alpha=0.7)
        ax_opt.plot(t_grid_hr, opt_components['core_rs'] * offset, color=color, ls='-.', lw=1.5, alpha=0.7)
        if include_wing and np.any(opt_components['wing'] > 0):
            ax_opt.plot(t_grid_hr, opt_components['wing'] * offset, color=color, ls='--', lw=1.5, alpha=0.7)
        if include_flare and np.any(opt_components['flare'] > 0):
            ax_opt.plot(t_grid_hr, opt_components['flare'] * offset, color=color, ls=(0, (5, 1)), lw=1.5, alpha=0.7)

    ax_opt.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_opt.set_ylabel('Flux density [mJy]', fontsize=11)
    ax_opt.set_xscale('log')
    ax_opt.set_yscale('log')
    handles, labels_text = ax_opt.get_legend_handles_labels()
    unique = dict(zip(labels_text, handles))
    ax_opt.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
    ax_opt.grid(which='both', alpha=0.25)
    ax_opt.set_title('Optical Light Curves', fontsize=12, fontweight='bold')

    # Set y-axis limits based on data
    set_log_y_limits(
        ax_xrt,
        xrt_data['flux'] - xrt_data['flux_error'],
        xrt_data['flux'] + xrt_data['flux_error'],
        xrt_components['total'],
    )

    opt_data_values = []
    for dataset in lc_data:
        offset = offset_factors.get(dataset['name'], 1.0)
        opt_data_values.extend((dataset['flux_mJy'] - dataset['flux_err']) * offset)
        opt_data_values.extend((dataset['flux_mJy'] + dataset['flux_err']) * offset)

    if opt_data_values:
        set_log_y_limits(ax_opt, *opt_data_values)

    # === 7DT Spectrum Panel ===
    if sdt_data:
        sdt_times = np.array([d['time'][0] for d in sdt_data])
        sdt_nu = np.array([d['frequency'] for d in sdt_data])
        sdt_wavelength = C_AA_PER_S / sdt_nu
        sdt_flux = np.array([d['flux_mJy'][0] for d in sdt_data])
        sdt_flux_err = np.array([d['flux_err'][0] for d in sdt_data])

        sed_time = float(np.median(sdt_times))
        wavelength_grid = np.geomspace(sdt_wavelength.min() * 0.9, sdt_wavelength.max() * 1.1, 300)
        nu_grid = C_AA_PER_S / wavelength_grid

        # Compute spectrum components
        sed_components_list = [compute_model_components(params, np.array([sed_time]), nu, None,
                                                        include_flare, include_wing)
                              for nu in nu_grid]

        total_sed = np.array([comp['total'][0] for comp in sed_components_list])
        fs_sed = np.array([comp['core_fs'][0] for comp in sed_components_list])
        rs_sed = np.array([comp['core_rs'][0] for comp in sed_components_list])
        wing_sed = np.array([comp['wing'][0] for comp in sed_components_list])

        # Plot 7DT data and model (with all components)
        ax_sed.errorbar(sdt_wavelength, sdt_flux, yerr=sdt_flux_err,
                       fmt='.', color='slategray', alpha=0.8, markersize=6, label='7DT data', zorder=5)
        ax_sed.plot(wavelength_grid, total_sed, 'k-', lw=2, label=f'Total at {sed_time/3600:.2f} hr', zorder=4)
        ax_sed.plot(wavelength_grid, fs_sed, 'k:', lw=1.5, label='Core FS', zorder=3)
        ax_sed.plot(wavelength_grid, rs_sed, 'c-.', lw=1.5, label='Reverse Shock', zorder=3)
        if include_wing and np.any(wing_sed > 0):
            ax_sed.plot(wavelength_grid, wing_sed, 'b--', lw=1.5, label='Wing', zorder=3)
        if include_flare:
            flare_sed = np.array([comp['flare'][0] for comp in sed_components_list])
            if np.any(flare_sed > 0):
                ax_sed.plot(wavelength_grid, flare_sed, color='red', ls=(0, (5, 1)),
                           lw=1.5, alpha=0.7, label='Flare', zorder=3)

        ax_sed.set_xlabel(r'Wavelength [$\AA$]', fontsize=11)
        ax_sed.set_ylabel('Flux density [mJy]', fontsize=11)
        ax_sed.set_xscale('log')
        ax_sed.set_yscale('log')
        ax_sed.legend(fontsize=8)
        ax_sed.grid(which='both', alpha=0.25)
        ax_sed.set_title('7DT Spectrum', fontsize=12, fontweight='bold')

        # Set y-axis limits for spectrum
        set_log_y_limits(
            ax_sed,
            sdt_flux - sdt_flux_err,
            sdt_flux + sdt_flux_err,
            total_sed,
        )

    fig.tight_layout()

    # Save
    outfile = outdir / "bestfit_lc.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {outfile}")

    return outfile


def plot_spectral_index_comparison(outdir):
    """Plot XRT spectral index evolution comparison"""
    outdir = Path(outdir)

    # Load best-fit parameters
    params, include_flare, include_wing = load_best_fit_params(outdir)

    # Try to load XRT spectral index data
    try:
        xrt_index_data = load_xrt_spectral_index()
    except Exception as e:
        print(f"  Warning: Could not load XRT spectral index data: {e}")
        return None

    if xrt_index_data is None:
        print("  Warning: No XRT spectral index data available")
        return None

    # Time grid for model
    t_grid = np.geomspace(100, 5e5, 200)  # 100s to ~6 days

    p = params['p']

    # Break frequencies for core and wing separately (panel 2)
    core_breaks = compute_break_frequencies(
        {"E_iso": params['E_iso_core'], "n_ism": params['n_ism'],
         "eps_e": params['eps_e'], "eps_B": params['eps_B'], "p": params['p']},
        REDSHIFT, t_grid
    )
    E_xrt_min = 0.3  # keV
    E_xrt_max = 10.0  # keV

    # Model spectral index from the SAME method the fit uses: local synchrotron
    # slope across the XRT band. Here we compute core and wing SEPARATELY (each
    # passed as the sole model, wing_model=None) so the two components are shown
    # as distinct expected curves rather than a summed slope.
    core_model = make_core_model(params)
    wing_model = make_wing_model(params) if (include_wing and "E_iso_wing" in params) else None

    # include_flare=False on the grid calls: no flare masking needed for the curves
    beta_core, _ = spectral_index_model(core_model, None, params, t_grid, False)
    photon_index_core = 1.0 - beta_core  # Gamma = 1 - beta

    if wing_model is not None:
        beta_wing, _ = spectral_index_model(wing_model, None, params, t_grid, False)
        photon_index_wing = 1.0 - beta_wing
        wing_breaks = compute_break_frequencies(
            {"E_iso": params['E_iso_wing'], "n_ism": params['n_ism'],
             "eps_e": params.get('eps_e_wing', params['eps_e']),
             "eps_B": params.get('eps_B_wing', params['eps_B']),
             "p": params.get('p_wing', params['p'])},
            REDSHIFT, t_grid
        )

    # keep mask = the fit's actual flux-contribution selection (uses summed core+wing)
    _, keep = spectral_index_model(core_model, wing_model, params,
                                   xrt_index_data['time'], include_flare)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Photon index
    t_grid_hr = t_grid / 3600
    t_data_hr = xrt_index_data['time'] / 3600
    gamma_obs = 1.0 - xrt_index_data['beta']
    yerr = np.array([xrt_index_data['beta_err_low'], xrt_index_data['beta_err_high']])

    ax1.errorbar(t_data_hr[keep], gamma_obs[keep], yerr=yerr[:, keep],
                fmt='o', color='black', markersize=4, alpha=0.8, label='XRT data (used in fit)')
    if np.any(~keep):
        ax1.errorbar(t_data_hr[~keep], gamma_obs[~keep], yerr=yerr[:, ~keep],
                    fmt='o', mfc='none', color='gray', markersize=4, alpha=0.6,
                    label='XRT data (excluded: flare-dominated)')
    ax1.plot(t_grid_hr, photon_index_core, 'r-', lw=2, label='Core model')
    if wing_model is not None:
        ax1.plot(t_grid_hr, photon_index_wing, 'b--', lw=2, label='Wing model')

    ax1.set_ylabel('Photon Index', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(which='both', alpha=0.25)
    ax1.set_title(f'XRT Spectral Index Evolution (p = {p:.3f})', fontsize=12, fontweight='bold')

    # Panel 2: Break frequencies (core solid, wing dashed)
    ax2.loglog(t_grid_hr, core_breaks["nu_m"], 'r-', lw=2, label=r'Core $\nu_m$')
    ax2.loglog(t_grid_hr, core_breaks["nu_c"], color='darkorange', lw=2, label=r'Core $\nu_c$')
    if wing_model is not None:
        ax2.loglog(t_grid_hr, wing_breaks["nu_m"], 'b--', lw=2, label=r'Wing $\nu_m$')
        ax2.loglog(t_grid_hr, wing_breaks["nu_c"], color='green', ls='--', lw=2, label=r'Wing $\nu_c$')

    # XRT band
    nu_xrt_min = E_xrt_min * 2.418e17  # keV to Hz
    nu_xrt_max = E_xrt_max * 2.418e17
    ax2.axhspan(nu_xrt_min, nu_xrt_max, alpha=0.2, color='gray', label='XRT band')

    ax2.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax2.set_ylabel('Frequency [Hz]', fontsize=11)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(which='both', alpha=0.25)
    ax2.set_title('Break Frequencies', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    outfile = outdir / "spectral_index_comparison.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {outfile}")

    return outfile
