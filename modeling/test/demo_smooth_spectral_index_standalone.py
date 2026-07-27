#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone demonstration of smooth spectral index transitions around breaks.

Shows that the spectral index changes smoothly (not sharply) as a function
of proximity to the cooling break, following Granot & Sari 2001.

This version doesn't require VegasAfterglow or other dependencies.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def flux_around_break(nu_ratio, beta1, beta2, s):
    """
    Compute flux around a spectral break using Granot & Sari 2001 Eq. 1.
    
    Fnu = Fnu_break * [(nu/nu_break)^(-s*beta1) + (nu/nu_break)^(-s*beta2)]^(-1/s)
    """
    x = np.asarray(nu_ratio)
    term1 = x ** (-s * beta1)
    term2 = x ** (-s * beta2)
    return (term1 + term2) ** (-1.0 / s)


def local_spectral_index(nu_ratio, beta1, beta2, s):
    """
    Compute local spectral index beta(nu) = d(ln F)/d(ln nu).
    
    This is the actual observable spectral index at a given frequency,
    accounting for the smooth transition around the break.
    
    Note: Following Granot & Sari convention where Fnu ~ nu^beta
    (beta is POSITIVE for rising spectra, NEGATIVE for declining spectra)
    """
    x = np.asarray(nu_ratio, dtype=float)
    
    # Numerical derivative
    dx = 1e-6
    f0 = flux_around_break(x, beta1, beta2, s)
    f1 = flux_around_break(x * (1.0 + dx), beta1, beta2, s)
    
    # beta = d(ln F)/d(ln nu)
    d_ln_f = np.log(f1 / f0)
    d_ln_nu = np.log(1.0 + dx)
    
    return d_ln_f / d_ln_nu


def sharpness_parameter_cooling(p):
    """
    Sharpness parameter s for the cooling break from Table 2.
    s(p) = 1.15 - 0.06*p
    """
    return 1.15 - 0.06 * p


def plot_smooth_transition():
    """Plot how spectral index varies smoothly around the cooling break."""
    
    p_values = [2.0, 2.2, 2.5, 2.8, 3.0]
    
    # Range around the break: 0.001 * nu_c to 1000 * nu_c  
    log_nu_ratios = np.linspace(-3, 3, 1000)
    nu_ratios = 10.0 ** log_nu_ratios
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Beta vs nu/nu_c for different p
    # Using Granot & Sari notation: Fnu ~ nu^beta_GS
    # Convert to X-ray convention: beta_X = -beta_GS where Fnu ~ nu^(-beta_X)
    for p in p_values:
        beta1_GS = (1.0 - p) / 2.0  # Below nu_c (Granot & Sari notation)
        beta2_GS = -p / 2.0           # Above nu_c (Granot & Sari notation)
        s = sharpness_parameter_cooling(p)
        
        beta_GS = local_spectral_index(nu_ratios, beta1_GS, beta2_GS, s)
        beta_X = -beta_GS  # Convert to X-ray convention
        
        ax1.semilogx(nu_ratios, beta_X, label=f'p = {p:.1f}', linewidth=2)
        
        # Show asymptotic values as dashed lines
        ax1.axhline(-beta1_GS, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax1.axhline(-beta2_GS, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.axvline(1.0, color='red', linestyle=':', alpha=0.5, linewidth=2, 
                label='nu_c (cooling break)')
    ax1.set_xlabel('nu / nu_c', fontsize=12)
    ax1.set_ylabel('Spectral index beta', fontsize=12)
    ax1.set_title('Smooth Spectral Index Transition Around Cooling Break\n' +
                  '(Granot & Sari 2001, Eq. 1)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(nu_ratios[0], nu_ratios[-1])
    
    # Plot 2: Comparison with step function approximation for p=2.2
    p_test = 2.2
    beta1_GS = (1.0 - p_test) / 2.0  # Granot & Sari notation
    beta2_GS = -p_test / 2.0
    s = sharpness_parameter_cooling(p_test)
    
    beta_smooth_GS = local_spectral_index(nu_ratios, beta1_GS, beta2_GS, s)
    beta_smooth = -beta_smooth_GS  # Convert to X-ray convention
    
    # Step function in X-ray convention
    beta1_X = -beta1_GS  # Below nu_c
    beta2_X = -beta2_GS  # Above nu_c
    beta_step = np.where(nu_ratios < 1.0, beta1_X, beta2_X)  # Sharp transition
    
    ax2.semilogx(nu_ratios, beta_smooth, 'b-', linewidth=2.5, 
                 label='Smooth (Granot & Sari 2001)')
    ax2.semilogx(nu_ratios, beta_step, 'r--', linewidth=2, 
                 label='Step function (simple approximation)')
    ax2.axvline(1.0, color='red', linestyle=':', alpha=0.5, linewidth=2)
    
    # Highlight transition region
    ax2.axvspan(0.5, 2.0, alpha=0.1, color='green', 
                label='Transition region\n(step function has error)')
    
    ax2.set_xlabel('nu / nu_c', fontsize=12)
    ax2.set_ylabel('Spectral index beta (X-ray convention)', fontsize=12)
    ax2.set_title(f'p = {p_test}: Smooth vs Step Function', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(nu_ratios[0], nu_ratios[-1])
    ax2.set_ylim(beta1_X - 0.1, beta2_X + 0.1)
    
    plt.tight_layout()
    plt.savefig('smooth_spectral_index.png', dpi=150, bbox_inches='tight')
    print("Saved: smooth_spectral_index.png")
    plt.close()


def demonstrate_interpolation_table():
    """Show the interpolation table concept."""
    print("=" * 80)
    print("Precomputed Spectral Index for Interpolation")
    print("=" * 80)
    print()
    print("NOTE: Using X-ray convention where Fnu ~ nu^(-beta), so:")
    print("  Below nu_c: beta = (p-1)/2  (softer)")
    print("  Above nu_c: beta = p/2      (harder)")
    print()
    
    # Build a simple lookup table
    p_grid = np.linspace(2.0, 3.0, 11)
    log_nu_ratio_grid = np.linspace(-2, 2, 41)
    
    print(f"Interpolation table dimensions:")
    print(f"  p values: {len(p_grid)} points from {p_grid[0]:.1f} to {p_grid[-1]:.1f}")
    print(f"  log10(nu/nu_c): {len(log_nu_ratio_grid)} points from {log_nu_ratio_grid[0]:.1f} to {log_nu_ratio_grid[-1]:.1f}")
    print(f"  Total table size: {len(p_grid)} x {len(log_nu_ratio_grid)} = {len(p_grid) * len(log_nu_ratio_grid)} values")
    print()
    
    # Compute a few example values
    print("Example values from the table:")
    print(f"{'p':>6s}  {'nu/nu_c':>10s}  {'beta (smooth)':>15s}  {'beta (step)':>12s}  {'Difference':>12s}")
    print("-" * 80)
    
    for p in [2.0, 2.2, 2.5, 2.8, 3.0]:
        for nu_ratio in [0.5, 1.0, 2.0]:
            # Granot & Sari notation
            beta1_GS = (1.0 - p) / 2.0
            beta2_GS = -p / 2.0
            s = sharpness_parameter_cooling(p)
            
            # Smooth transition
            beta_smooth_GS = local_spectral_index(nu_ratio, beta1_GS, beta2_GS, s)
            beta_smooth = -beta_smooth_GS  # X-ray convention
            
            # Step function (X-ray convention)
            beta1_X = -beta1_GS
            beta2_X = -beta2_GS
            beta_step = beta1_X if nu_ratio < 1.0 else beta2_X
            diff = beta_smooth - beta_step
            
            print(f"{p:>6.1f}  {nu_ratio:>10.1f}  {beta_smooth:>15.4f}  {beta_step:>12.4f}  {diff:>12.4f}")
    
    print()
    print("During fitting, you can interpolate these precomputed values")
    print("instead of assuming a sharp step function.")
    print()


def main():
    """Run demonstrations."""
    plot_smooth_transition()
    print()
    demonstrate_interpolation_table()
    print()
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    print("1. The spectral index changes SMOOTHLY around break frequencies,")
    print("   not as a sharp step function.")
    print()
    print("2. If nu_obs is near nu_c (within factor of ~3), the transition")
    print("   matters significantly for accurate spectral index predictions.")
    print()
    print("3. For fitting: precompute beta(p, nu/nu_c) on a grid and use")
    print("   2D interpolation to get the spectral index quickly.")
    print()
    print("4. This removes the need to assume 'slow' or 'fast' cooling")
    print("   regime - the smooth formula works in all cases!")
    print("=" * 80)


if __name__ == "__main__":
    main()
