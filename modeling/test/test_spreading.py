#!/usr/bin/env python3
"""
Test the effect of turning off spreading in the core jet
Compares spreading=True (current) vs spreading=False
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet
from VegasAfterglow.units import keV

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from grb.const import D_L, REDSHIFT
from grb.io import read_data
from grb.extinction import host_extinction_attenuation
from grb.utils import flux_error, model_array

# XRT band
XRT_BAND = (0.3 * keV, 10.0 * keV)
MODEL_RESOLUTIONS = (0.1, 0.25, 10)

# Best-fit parameters from final_flare_wing_20260714_024955
params = {
    'E_iso_core': 6.5652e+51,
    'Gamma0_core': 578.398,
    'theta_c_core': 0.0598068,
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
}


def make_core_model_with_spreading(params, spreading=True):
    """Create core jet model with or without spreading"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_core"],
        Gamma0=params["Gamma0_core"],
        theta_c=params["theta_c_core"],
        spreading=spreading,  # Toggle spreading here
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


def main():
    print("=" * 60)
    print("Testing Effect of Turning Off Spreading")
    print("=" * 60)
    print("\nLoading data...")
    
    # Load XRT and optical data
    xrt_data = read_data("xrt")
    i_data = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    
    # Create time grid
    t_min = min(xrt_data['time'].min(), i_data['time'].min())
    t_max = max(xrt_data['time'].max(), i_data['time'].max())
    t_grid = np.geomspace(t_min * 0.8, t_max * 1.2, 300)
    
    print("\nComputing models...")
    
    # Model WITH spreading (current)
    print("  - With spreading=True (current)")
    model_with = make_core_model_with_spreading(params, spreading=True)
    
    xrt_with = model_with.flux(t_grid, XRT_BAND[0], XRT_BAND[1], 10)
    xrt_flux_with = model_array(xrt_with.total)
    xrt_fwd_with = model_array(xrt_with.fwd)
    xrt_rvs_with = model_array(xrt_with.rvs)
    
    i_nu = float(i_data['frequency_Hz'].iloc[0])
    i_with = model_with.flux_density(t_grid, i_nu * np.ones_like(t_grid))
    i_flux_with = model_array(i_with.total)
    i_fwd_with = model_array(i_with.fwd)
    i_rvs_with = model_array(i_with.rvs)
    
    # Apply extinction
    attenuation = host_extinction_attenuation(i_nu * np.ones_like(t_grid), params['A_V'], REDSHIFT)
    i_flux_with = i_flux_with * attenuation * 1e26
    i_fwd_with = i_fwd_with * attenuation * 1e26
    i_rvs_with = i_rvs_with * attenuation * 1e26
    
    # Model WITHOUT spreading
    print("  - With spreading=False (test)")
    model_without = make_core_model_with_spreading(params, spreading=False)
    
    xrt_without = model_without.flux(t_grid, XRT_BAND[0], XRT_BAND[1], 10)
    xrt_flux_without = model_array(xrt_without.total)
    xrt_fwd_without = model_array(xrt_without.fwd)
    xrt_rvs_without = model_array(xrt_without.rvs)
    
    i_without = model_without.flux_density(t_grid, i_nu * np.ones_like(t_grid))
    i_flux_without = model_array(i_without.total)
    i_fwd_without = model_array(i_without.fwd)
    i_rvs_without = model_array(i_without.rvs)
    
    # Apply extinction
    i_flux_without = i_flux_without * attenuation * 1e26
    i_fwd_without = i_fwd_without * attenuation * 1e26
    i_rvs_without = i_rvs_without * attenuation * 1e26
    
    print("\nPlotting comparison...")
    
    # Create comparison plot
    fig, ((ax_xrt, ax_i), (ax_ratio_xrt, ax_ratio_i)) = plt.subplots(2, 2, figsize=(14, 10))
    
    t_grid_hr = t_grid / 3600
    xrt_err = flux_error(xrt_data)
    
    # XRT panel
    ax_xrt.errorbar(xrt_data['time'] / 3600, xrt_data['flux'], yerr=xrt_err,
                    fmt='.', color='black', alpha=0.7, markersize=6, label='XRT data')
    ax_xrt.plot(t_grid_hr, xrt_flux_with, 'b-', lw=2, label='With spreading (current)')
    ax_xrt.plot(t_grid_hr, xrt_fwd_with, 'b:', lw=1.5, alpha=0.7, label='FS (spreading)')
    ax_xrt.plot(t_grid_hr, xrt_rvs_with, 'b-.', lw=1.5, alpha=0.7, label='RS (spreading)')
    
    ax_xrt.plot(t_grid_hr, xrt_flux_without, 'r-', lw=2, label='Without spreading')
    ax_xrt.plot(t_grid_hr, xrt_fwd_without, 'r:', lw=1.5, alpha=0.7, label='FS (no spreading)')
    ax_xrt.plot(t_grid_hr, xrt_rvs_without, 'r-.', lw=1.5, alpha=0.7, label='RS (no spreading)')
    
    ax_xrt.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_xrt.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=11)
    ax_xrt.set_xscale('log')
    ax_xrt.set_yscale('log')
    ax_xrt.legend(fontsize=8, ncol=2)
    ax_xrt.grid(which='both', alpha=0.25)
    ax_xrt.set_title('XRT 0.3-10 keV', fontsize=12, fontweight='bold')
    
    # i-band panel
    ax_i.errorbar(i_data['time'] / 3600, i_data['flux_mJy'], yerr=i_data['flux_mJy_error'],
                  fmt='.', color='black', alpha=0.7, markersize=6, label='i-band data')
    ax_i.plot(t_grid_hr, i_flux_with, 'b-', lw=2, label='With spreading (current)')
    ax_i.plot(t_grid_hr, i_fwd_with, 'b:', lw=1.5, alpha=0.7, label='FS (spreading)')
    ax_i.plot(t_grid_hr, i_rvs_with, 'b-.', lw=1.5, alpha=0.7, label='RS (spreading)')
    
    ax_i.plot(t_grid_hr, i_flux_without, 'r-', lw=2, label='Without spreading')
    ax_i.plot(t_grid_hr, i_fwd_without, 'r:', lw=1.5, alpha=0.7, label='FS (no spreading)')
    ax_i.plot(t_grid_hr, i_rvs_without, 'r-.', lw=1.5, alpha=0.7, label='RS (no spreading)')
    
    ax_i.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_i.set_ylabel('Flux density [mJy]', fontsize=11)
    ax_i.set_xscale('log')
    ax_i.set_yscale('log')
    ax_i.legend(fontsize=8, ncol=2)
    ax_i.grid(which='both', alpha=0.25)
    ax_i.set_title('i-band', fontsize=12, fontweight='bold')
    
    # Ratio panels
    ratio_xrt = xrt_flux_without / xrt_flux_with
    ratio_i = i_flux_without / i_flux_with
    
    ax_ratio_xrt.plot(t_grid_hr, ratio_xrt, 'k-', lw=2)
    ax_ratio_xrt.axhline(1, color='gray', ls='--', lw=1)
    ax_ratio_xrt.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_ratio_xrt.set_ylabel('Ratio (no spread / spread)', fontsize=11)
    ax_ratio_xrt.set_xscale('log')
    ax_ratio_xrt.grid(which='both', alpha=0.25)
    ax_ratio_xrt.set_title('XRT Flux Ratio', fontsize=11)
    
    ax_ratio_i.plot(t_grid_hr, ratio_i, 'k-', lw=2)
    ax_ratio_i.axhline(1, color='gray', ls='--', lw=1)
    ax_ratio_i.set_xlabel('Time since trigger [hr]', fontsize=11)
    ax_ratio_i.set_ylabel('Ratio (no spread / spread)', fontsize=11)
    ax_ratio_i.set_xscale('log')
    ax_ratio_i.grid(which='both', alpha=0.25)
    ax_ratio_i.set_title('i-band Flux Ratio', fontsize=11)
    
    plt.tight_layout()
    
    # Save to current directory
    import os
    outfile = os.path.join(os.path.dirname(__file__), 'spreading_comparison.png')
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {outfile}")
    except OSError as e:
        print(f"\nNote: Could not save plot ({e})")
        print("  Run the script directly to save the plot")
    
    # Print some statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    # Find when ratio differs most
    idx_max_diff_xrt = np.argmax(np.abs(ratio_xrt - 1))
    idx_max_diff_i = np.argmax(np.abs(ratio_i - 1))
    
    print(f"\nXRT:")
    print(f"  Max difference at t = {t_grid_hr[idx_max_diff_xrt]:.2f} hr")
    print(f"  Ratio = {ratio_xrt[idx_max_diff_xrt]:.3f}")
    print(f"  With spreading: {xrt_flux_with[idx_max_diff_xrt]:.3e} erg/cm²/s")
    print(f"  Without spreading: {xrt_flux_without[idx_max_diff_xrt]:.3e} erg/cm²/s")
    
    print(f"\ni-band:")
    print(f"  Max difference at t = {t_grid_hr[idx_max_diff_i]:.2f} hr")
    print(f"  Ratio = {ratio_i[idx_max_diff_i]:.3f}")
    print(f"  With spreading: {i_flux_with[idx_max_diff_i]:.3e} mJy")
    print(f"  Without spreading: {i_flux_without[idx_max_diff_i]:.3e} mJy")
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
