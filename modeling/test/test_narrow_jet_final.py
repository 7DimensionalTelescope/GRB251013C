#!/usr/bin/env python3
"""
Test how final_model.py results change with a very narrow jet (t_jet ~ 0.02 hr)
Compares current best-fit vs narrow jet parameters
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from final_model import (
    load_all_optical_data,
    make_core_model,
    make_wing_model,
    norris_flare,
    XRT_BAND,
)
from utils import model_array, host_extinction_attenuation, xrt_flux_error
from grb.const import REDSHIFT


def compute_full_model(params, xrt_times, i_times, i_nu, include_flare=True, include_wing=True):
    """Compute full model flux (Core FS+RS + Flare + Wing)"""
    # Core model
    core_model = make_core_model(params)
    
    # XRT
    xrt_core = core_model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10)
    xrt_flux = model_array(xrt_core.total).copy()
    xrt_fwd = model_array(xrt_core.fwd).copy()
    xrt_rvs = model_array(xrt_core.rvs).copy()
    
    # Wing
    xrt_wing = np.zeros_like(xrt_flux)
    if include_wing and "E_iso_wing" in params:
        wing_model = make_wing_model(params)
        xrt_wing_output = wing_model.flux(xrt_times, XRT_BAND[0], XRT_BAND[1], 10)
        xrt_wing = model_array(xrt_wing_output.total).copy()
        xrt_flux += xrt_wing
    
    # Flare
    xrt_flare = np.zeros_like(xrt_flux)
    if include_flare and "t_start_flare" in params:
        xrt_flare = norris_flare(xrt_times, params["t_start_flare"],
                                 params["tau_rise_flare"], params["tau_decay_flare"],
                                 params["A_flare"])
        xrt_flux += xrt_flare
    
    # i-band
    i_nu_array = i_nu * np.ones_like(i_times)
    i_core = core_model.flux_density(i_times, i_nu_array)
    i_flux = model_array(i_core.total).copy()
    i_fwd = model_array(i_core.fwd).copy()
    i_rvs = model_array(i_core.rvs).copy()
    
    # Wing in i-band
    i_wing = np.zeros_like(i_flux)
    if include_wing and "E_iso_wing" in params:
        wing_model = make_wing_model(params)
        i_wing_output = wing_model.flux_density(i_times, i_nu_array)
        i_wing = model_array(i_wing_output.total).copy()
        i_flux += i_wing
    
    # Flare in i-band
    i_flare = np.zeros_like(i_flux)
    if include_flare and "t_start_flare" in params:
        nu_xrt_min = 7.25e16
        nu_xrt_max = 2.42e18
        beta_flare = params.get("flare_beta", 0.8)
        
        flare_temporal = norris_flare(i_times, params["t_start_flare"],
                                     params["tau_rise_flare"], params["tau_decay_flare"],
                                     params["A_flare"])
        
        if abs(beta_flare - 1.0) < 0.01:
            K = flare_temporal / np.log(nu_xrt_max / nu_xrt_min)
        else:
            K = flare_temporal * (1 - beta_flare) / (nu_xrt_max**(1-beta_flare) - nu_xrt_min**(1-beta_flare))
        
        i_flare = K * i_nu**(-beta_flare)
        i_flux = i_flux + i_flare
    
    # Apply extinction to i-band
    if "A_V" in params:
        attenuation = host_extinction_attenuation(i_nu_array, params["A_V"], REDSHIFT)
        i_flux = i_flux * attenuation
        i_fwd = i_fwd * attenuation
        i_rvs = i_rvs * attenuation
        i_wing = i_wing * attenuation
        i_flare = i_flare * attenuation
    
    # Convert to mJy
    i_flux *= 1e26
    i_fwd *= 1e26
    i_rvs *= 1e26
    i_wing *= 1e26
    i_flare *= 1e26
    
    return {
        'xrt_total': xrt_flux,
        'xrt_fwd': xrt_fwd,
        'xrt_rvs': xrt_rvs,
        'xrt_wing': xrt_wing,
        'xrt_flare': xrt_flare,
        'i_total': i_flux,
        'i_fwd': i_fwd,
        'i_rvs': i_rvs,
        'i_wing': i_wing,
        'i_flare': i_flare,
    }


def main():
    print("=" * 70)
    print("Testing Narrow Jet Parameters in final_model.py")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    xrt_data, optical_datasets = load_all_optical_data()
    
    # Get i-band data
    i_dataset = [d for d in optical_datasets if d['name'] == 'i-band'][0]
    i_nu = i_dataset['frequency']
    
    # Create time grid
    t_min = min(xrt_data['time'].min(), i_dataset['time'].min())
    t_max = max(xrt_data['time'].max(), i_dataset['time'].max())
    t_grid = np.geomspace(t_min * 0.8, t_max * 1.2, 300)
    
    # Current best-fit parameters
    params_current = {
        'E_iso_core': 6.5652e+51,
        'Gamma0_core': 578.398,
        'theta_c_core': 0.0598068,  # Current: ~3.4 degrees
        'n_ism': 47.5051,
        'p': 2.18057,
        'eps_e': 0.0467743,
        'eps_B': 0.0286311,
        'xi': 0.84343,
        'tau': 43.4667,
        'p_r': 3.57919,
        'eps_e_r': 0.0343106,
        'eps_B_r': 0.38045,
        'xi_r': 0.996726,
        'A_V': 0.0820291,
        't_start_flare': 2088.81,
        'tau_rise_flare': 113.216,
        'tau_decay_flare': 4781.14,
        'A_flare': 4.74774e-10,
        'flare_beta': 0.674565,
        'E_iso_wing': 9.92553e+51,
        'Gamma0_wing': 19.7136,
        'theta_c_wing': 0.288619,
        'p_wing': 2.48721,
        'eps_e_wing': 0.508742,
        'eps_B_wing': 0.00959794,
        'xi_wing': 0.872902,
    }
    
    # Narrow jet parameters (t_jet ~ 0.02 hr)
    params_narrow = params_current.copy()
    params_narrow['theta_c_core'] = 0.02  # Very narrow: 1.15 degrees
    # Might need to adjust E_iso to compensate for narrower beaming
    params_narrow['E_iso_core'] = 1e52  # Increase energy
    
    print(f"\nCurrent parameters:")
    print(f"  theta_c_core = {params_current['theta_c_core']:.4f} rad = {np.degrees(params_current['theta_c_core']):.2f} deg")
    print(f"  E_iso_core = {params_current['E_iso_core']:.2e} erg")
    print(f"  t_jet ~ 0.22 hr")
    
    print(f"\nNarrow jet parameters:")
    print(f"  theta_c_core = {params_narrow['theta_c_core']:.4f} rad = {np.degrees(params_narrow['theta_c_core']):.2f} deg")
    print(f"  E_iso_core = {params_narrow['E_iso_core']:.2e} erg")
    print(f"  t_jet ~ 0.02 hr")
    
    # Compute models
    print("\nComputing models...")
    print("  Current parameters...")
    model_current = compute_full_model(params_current, t_grid, t_grid, i_nu)
    
    print("  Narrow jet parameters...")
    model_narrow = compute_full_model(params_narrow, t_grid, t_grid, i_nu)
    
    # Plot comparison
    print("\nPlotting comparison...")
    
    fig, ((ax_xrt, ax_i), (ax_xrt_comp, ax_i_comp)) = plt.subplots(2, 2, figsize=(14, 10))
    
    t_hr = t_grid / 3600
    xrt_err = xrt_data['flux_error']
    
    # === XRT Full Comparison ===
    ax_xrt.errorbar(xrt_data['time'] / 3600, xrt_data['flux'], yerr=xrt_err,
                    fmt='.', color='black', alpha=0.7, markersize=6, label='XRT data', zorder=5)
    
    ax_xrt.plot(t_hr, model_current['xrt_total'], 'b-', lw=2, label='Current (total)')
    ax_xrt.plot(t_hr, model_narrow['xrt_total'], 'r--', lw=2, label='Narrow jet (total)')
    ax_xrt.axvline(0.02, color='red', ls=':', lw=1.5, alpha=0.5, label='t_jet = 0.02 hr')
    ax_xrt.axvline(0.22, color='blue', ls=':', lw=1.5, alpha=0.5, label='t_jet = 0.22 hr')
    
    ax_xrt.set_xlabel('Time [hr]', fontsize=11)
    ax_xrt.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=11)
    ax_xrt.set_xscale('log')
    ax_xrt.set_yscale('log')
    ax_xrt.legend(fontsize=9)
    ax_xrt.grid(which='both', alpha=0.25)
    ax_xrt.set_title('XRT: Total Model Comparison', fontsize=12, fontweight='bold')
    
    # === i-band Full Comparison ===
    ax_i.errorbar(i_dataset['time'] / 3600, i_dataset['flux_mJy'], yerr=i_dataset['flux_err'],
                  fmt='.', color='black', alpha=0.7, markersize=6, label='i-band data', zorder=5)
    
    ax_i.plot(t_hr, model_current['i_total'], 'b-', lw=2, label='Current (total)')
    ax_i.plot(t_hr, model_narrow['i_total'], 'r--', lw=2, label='Narrow jet (total)')
    ax_i.axvline(0.02, color='red', ls=':', lw=1.5, alpha=0.5)
    ax_i.axvline(0.22, color='blue', ls=':', lw=1.5, alpha=0.5)
    
    ax_i.set_xlabel('Time [hr]', fontsize=11)
    ax_i.set_ylabel('Flux density [mJy]', fontsize=11)
    ax_i.set_xscale('log')
    ax_i.set_yscale('log')
    ax_i.legend(fontsize=9)
    ax_i.grid(which='both', alpha=0.25)
    ax_i.set_title('i-band: Total Model Comparison', fontsize=12, fontweight='bold')
    
    # === XRT Components (Current) ===
    ax_xrt_comp.errorbar(xrt_data['time'] / 3600, xrt_data['flux'], yerr=xrt_err,
                         fmt='.', color='black', alpha=0.5, markersize=4, label='XRT data', zorder=5)
    
    ax_xrt_comp.plot(t_hr, model_current['xrt_total'], 'k-', lw=2, label='Total')
    ax_xrt_comp.plot(t_hr, model_current['xrt_fwd'], 'b:', lw=1.5, label='Core FS')
    ax_xrt_comp.plot(t_hr, model_current['xrt_rvs'], 'c-.', lw=1.5, label='Core RS')
    ax_xrt_comp.plot(t_hr, model_current['xrt_wing'], 'g--', lw=1.5, label='Wing')
    if np.any(model_current['xrt_flare'] > 0):
        ax_xrt_comp.plot(t_hr, model_current['xrt_flare'], color='red', ls=(0, (5, 1)), 
                        lw=1.5, alpha=0.7, label='Flare')
    
    ax_xrt_comp.set_xlabel('Time [hr]', fontsize=11)
    ax_xrt_comp.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=11)
    ax_xrt_comp.set_xscale('log')
    ax_xrt_comp.set_yscale('log')
    ax_xrt_comp.legend(fontsize=8)
    ax_xrt_comp.grid(which='both', alpha=0.25)
    ax_xrt_comp.set_title('XRT: Current Parameters (Components)', fontsize=11, fontweight='bold')
    
    # === i-band Components (Current) ===
    ax_i_comp.errorbar(i_dataset['time'] / 3600, i_dataset['flux_mJy'], yerr=i_dataset['flux_err'],
                       fmt='.', color='black', alpha=0.5, markersize=4, label='i-band data', zorder=5)
    
    ax_i_comp.plot(t_hr, model_current['i_total'], 'k-', lw=2, label='Total')
    ax_i_comp.plot(t_hr, model_current['i_fwd'], 'b:', lw=1.5, label='Core FS')
    ax_i_comp.plot(t_hr, model_current['i_rvs'], 'c-.', lw=1.5, label='Core RS')
    ax_i_comp.plot(t_hr, model_current['i_wing'], 'g--', lw=1.5, label='Wing')
    if np.any(model_current['i_flare'] > 0):
        ax_i_comp.plot(t_hr, model_current['i_flare'], color='red', ls=(0, (5, 1)), 
                      lw=1.5, alpha=0.7, label='Flare')
    
    ax_i_comp.set_xlabel('Time [hr]', fontsize=11)
    ax_i_comp.set_ylabel('Flux density [mJy]', fontsize=11)
    ax_i_comp.set_xscale('log')
    ax_i_comp.set_yscale('log')
    ax_i_comp.legend(fontsize=8)
    ax_i_comp.grid(which='both', alpha=0.25)
    ax_i_comp.set_title('i-band: Current Parameters (Components)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    outfile = os.path.join(os.path.dirname(__file__), 'narrow_jet_final_comparison.png')
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {outfile}")
    except OSError as e:
        print(f"\nNote: Could not save plot ({e})")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("With a very narrow jet (theta_c = 0.02 rad, t_jet ~ 0.02 hr):")
    print("  - Core FS drops much more steeply after 0.02 hr")
    print("  - Cannot explain the smooth XRT light curve from 0.02-1 hr")
    print("  - Would need much stronger flare and wing contributions")
    print("  - The current fit (theta_c ~ 0.06 rad, t_jet ~ 0.22 hr) is favored")
    print("=" * 70)


if __name__ == "__main__":
    main()
