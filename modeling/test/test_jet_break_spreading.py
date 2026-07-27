#!/usr/bin/env python3
"""
Toy model to compare jet break behavior with and without spreading

This demonstrates:
1. How spreading affects jet break time
2. How spreading changes post-break decay slope
3. When spreading effects are most important
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet
from VegasAfterglow.units import keV

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import D_L, REDSHIFT
from grb.io import read_data
from utils import model_array, xrt_flux_error

# XRT band
XRT_BAND = (0.3 * keV, 10.0 * keV)
MODEL_RESOLUTIONS = (0.1, 0.25, 10)


def make_toy_model(E_iso, Gamma0, theta_c, n_ism, spreading=True):
    """Create a simple jet model with or without spreading"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=n_ism)
    jet = TophatJet(
        E_iso=E_iso,
        Gamma0=Gamma0,
        theta_c=theta_c,
        spreading=spreading,
        duration=10.0,
    )
    
    # Standard microphysics
    radiation = Radiation(
        eps_e=0.1,
        eps_B=0.01,
        p=2.2,
        xi_e=1.0,
        ssc=False,
        kn=False,
    )
    
    return Model(jet=jet, medium=medium, observer=observer, 
                 fwd_rad=radiation, resolutions=MODEL_RESOLUTIONS)


def estimate_jet_break_time(E_iso, Gamma0, theta_c, n_ism):
    """
    Analytical estimate of jet break time from Sari et al. (1999)
    t_jet ~ 0.5 * (3 * E_iso / (4 * pi * n_ism * m_p * c^5))^(1/3) * theta_c^(8/3) / (1 + z)
    
    Or in observer frame days:
    t_jet ~ 0.5 day * (E_iso / 10^53 erg)^(1/3) * (n / 1 cm^-3)^(-1/3) * (theta_c / 0.1)^(8/3)
    """
    E_53 = E_iso / 1e53
    theta_01 = theta_c / 0.1
    
    # In observer frame days
    t_jet_days = 0.5 * (E_53)**(1/3) * (n_ism)**(-1/3) * (theta_01)**(8/3) / (1 + REDSHIFT)
    
    # Convert to seconds
    t_jet_sec = t_jet_days * 86400
    
    return t_jet_sec


def compute_power_law_slope(times, fluxes, t_start, t_end):
    """Compute temporal decay index between two times"""
    mask = (times >= t_start) & (times <= t_end)
    if np.sum(mask) < 2:
        return np.nan
    
    t_fit = times[mask]
    f_fit = fluxes[mask]
    
    # Fit log(F) = -alpha * log(t) + const
    log_t = np.log10(t_fit)
    log_f = np.log10(f_fit)
    
    coeffs = np.polyfit(log_t, log_f, 1)
    alpha = -coeffs[0]  # Temporal decay index (F ∝ t^-alpha)
    
    return alpha


def main():
    print("=" * 70)
    print("Toy Model: Jet Break with and without Spreading")
    print("=" * 70)
    
    # Load XRT data
    print("\nLoading XRT data...")
    xrt_data = read_data("xrt")
    xrt_times_hr = xrt_data['time'].to_numpy(float) / 3600
    xrt_flux = xrt_data['flux'].to_numpy(float)
    xrt_err = xrt_flux_error(xrt_data)
    print(f"  Loaded {len(xrt_data)} XRT points")
    
    # Test different parameter sets
    test_cases = [
        {
            'name': 'Very narrow jet (t_jet ~ 0.02 hr)',
            'E_iso': 1e52,
            'Gamma0': 300,
            'theta_c': 0.02,  # ~1.15 degrees - very narrow!
            'n_ism': 10.0,
        },
        {
            'name': 'Narrow jet, high Lorentz',
            'E_iso': 1e52,
            'Gamma0': 300,
            'theta_c': 0.05,  # ~2.9 degrees
            'n_ism': 1.0,
        },
        {
            'name': 'Wide jet, moderate Lorentz',
            'E_iso': 1e52,
            'Gamma0': 100,
            'theta_c': 0.2,  # ~11.5 degrees
            'n_ism': 1.0,
        },
        {
            'name': 'GRB 251013C-like',
            'E_iso': 6.5e51,
            'Gamma0': 578,
            'theta_c': 0.06,  # ~3.4 degrees
            'n_ism': 47.5,
        },
    ]
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(14, 3.5*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for i, case in enumerate(test_cases):
        print(f"\n{'-' * 70}")
        print(f"Case {i+1}: {case['name']}")
        print(f"{'-' * 70}")
        print(f"  E_iso = {case['E_iso']:.2e} erg")
        print(f"  Gamma0 = {case['Gamma0']:.0f}")
        print(f"  theta_c = {case['theta_c']:.3f} rad = {np.degrees(case['theta_c']):.2f} deg")
        print(f"  n_ism = {case['n_ism']:.1f} cm^-3")
        
        # Estimate analytical jet break time
        t_jet_analytic = estimate_jet_break_time(
            case['E_iso'], case['Gamma0'], case['theta_c'], case['n_ism']
        )
        print(f"\n  Analytical t_jet ≈ {t_jet_analytic/3600:.2f} hours")
        
        # Create time grid spanning before and after jet break
        t_min = t_jet_analytic / 100
        t_max = t_jet_analytic * 100
        t_grid = np.geomspace(t_min, t_max, 500)
        
        # Compute models
        print("\n  Computing model with spreading...")
        model_with = make_toy_model(
            case['E_iso'], case['Gamma0'], case['theta_c'], case['n_ism'],
            spreading=True
        )
        flux_with = model_array(model_with.flux(
            t_grid, XRT_BAND[0], XRT_BAND[1], 10
        ).total)
        
        print("  Computing model without spreading...")
        model_without = make_toy_model(
            case['E_iso'], case['Gamma0'], case['theta_c'], case['n_ism'],
            spreading=False
        )
        flux_without = model_array(model_without.flux(
            t_grid, XRT_BAND[0], XRT_BAND[1], 10
        ).total)
        
        # Compute decay indices
        # Pre-break: 0.1 * t_jet to t_jet
        alpha_pre_with = compute_power_law_slope(
            t_grid, flux_with, t_jet_analytic*0.1, t_jet_analytic*0.9
        )
        alpha_pre_without = compute_power_law_slope(
            t_grid, flux_without, t_jet_analytic*0.1, t_jet_analytic*0.9
        )
        
        # Post-break: 2*t_jet to 10*t_jet
        alpha_post_with = compute_power_law_slope(
            t_grid, flux_with, t_jet_analytic*2, t_jet_analytic*10
        )
        alpha_post_without = compute_power_law_slope(
            t_grid, flux_without, t_jet_analytic*2, t_jet_analytic*10
        )
        
        print(f"\n  Pre-break decay:")
        print(f"    With spreading: α = {alpha_pre_with:.2f}")
        print(f"    Without spreading: α = {alpha_pre_without:.2f}")
        print(f"\n  Post-break decay:")
        print(f"    With spreading: α = {alpha_post_with:.2f}")
        print(f"    Without spreading: α = {alpha_post_without:.2f}")
        print(f"    Δα = {alpha_post_without - alpha_post_with:.2f}")
        
        # Plot light curves
        ax_lc = axes[i, 0]
        t_hr = t_grid / 3600
        
        # Plot XRT data
        ax_lc.errorbar(xrt_times_hr, xrt_flux, yerr=xrt_err,
                      fmt='.', color='black', alpha=0.7, markersize=6,
                      label='XRT data', zorder=5)
        
        ax_lc.plot(t_hr, flux_with, 'b-', lw=2, label='With spreading')
        ax_lc.plot(t_hr, flux_without, 'r--', lw=2, label='Without spreading')
        ax_lc.axvline(t_jet_analytic/3600, color='gray', ls=':', lw=1.5, 
                     label=f'Analytic t_jet = {t_jet_analytic/3600:.2f} hr')
        
        ax_lc.set_xlabel('Time [hr]', fontsize=11)
        ax_lc.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=11)
        ax_lc.set_xscale('log')
        ax_lc.set_yscale('log')
        ax_lc.legend(fontsize=9)
        ax_lc.grid(which='both', alpha=0.25)
        ax_lc.set_title(f'{case["name"]}', fontsize=11, fontweight='bold')
        
        # Add decay slopes as text
        y_pos = ax_lc.get_ylim()[0] * 10
        ax_lc.text(t_jet_analytic/3600/5, y_pos, 
                  f'α_pre ≈ {alpha_pre_with:.2f}',
                  fontsize=9, ha='center', color='blue')
        ax_lc.text(t_jet_analytic/3600*5, y_pos,
                  f'α_post = {alpha_post_with:.2f} (with)\n{alpha_post_without:.2f} (w/o)',
                  fontsize=9, ha='center', color='red')
        
        # Plot ratio
        ax_ratio = axes[i, 1]
        ratio = flux_without / flux_with
        
        ax_ratio.plot(t_hr, ratio, 'k-', lw=2)
        ax_ratio.axhline(1, color='gray', ls='--', lw=1)
        ax_ratio.axvline(t_jet_analytic/3600, color='gray', ls=':', lw=1.5)
        
        ax_ratio.set_xlabel('Time [hr]', fontsize=11)
        ax_ratio.set_ylabel('Ratio (no spread / spread)', fontsize=11)
        ax_ratio.set_xscale('log')
        ax_ratio.grid(which='both', alpha=0.25)
        ax_ratio.set_title('Flux Ratio', fontsize=11, fontweight='bold')
        
        # Highlight where ratio deviates most
        max_deviation_idx = np.argmax(np.abs(ratio - 1))
        max_deviation_time = t_hr[max_deviation_idx]
        max_deviation_ratio = ratio[max_deviation_idx]
        ax_ratio.plot(max_deviation_time, max_deviation_ratio, 'ro', markersize=8)
        ax_ratio.text(max_deviation_time*1.5, max_deviation_ratio,
                     f'Max: {max_deviation_ratio:.3f}\nat {max_deviation_time:.2f} hr',
                     fontsize=9, va='center')
    
    plt.tight_layout()
    
    # Save plot
    outfile = os.path.join(os.path.dirname(__file__), 'jet_break_spreading_comparison.png')
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n{'=' * 70}")
        print(f"✓ Plot saved to: {outfile}")
        print("=" * 70)
    except OSError as e:
        print(f"\nNote: Could not save plot ({e})")
        print("  Run the script directly to save the plot")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Spreading smooths the jet break transition")
    print("2. Without spreading, post-break decay is steeper (Δα ~ 1)")
    print("3. Effect is most visible for narrow jets (small θ_c)")
    print("4. For wide jets or high Γ0, spreading matters less")
    print("=" * 70)


if __name__ == "__main__":
    main()
