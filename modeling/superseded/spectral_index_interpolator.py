#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precomputed spectral index interpolator based on Granot & Sari 2001.

The spectral breaks are smooth, not sharp. The local spectral index beta(nu)
changes continuously as a function of nu/nu_break. This module precomputes
the relationship and provides fast interpolation for fitting.

Reference: Granot & Sari 2001, arXiv:astro-ph/0108027v1, Equation 1
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def flux_around_break(nu_ratio, beta1, beta2, s):
    """
    Compute flux around a spectral break using Granot & Sari 2001 Eq. 1.
    
    Args:
        nu_ratio: nu/nu_break (can be array)
        beta1: spectral slope below break
        beta2: spectral slope above break  
        s: sharpness parameter
        
    Returns:
        Normalized flux (Fnu / Fnu_break_ext)
    """
    x = np.asarray(nu_ratio)
    term1 = x ** (-s * beta1)
    term2 = x ** (-s * beta2)
    return (term1 + term2) ** (-1.0 / s)


def local_spectral_index(nu_ratio, beta1, beta2, s):
    """
    Compute local spectral index beta(nu) = d(ln F)/d(ln nu).
    
    For the smooth break given by Granot & Sari Eq. 1, we compute
    the derivative numerically to get the local slope.
    
    Note: Granot & Sari define beta such that F_nu ~ nu^beta (positive exponent).
    This matches the X-ray convention where photon index Gamma = beta + 1.
    
    Args:
        nu_ratio: nu/nu_break (can be array)
        beta1: spectral slope below break (nu << nu_break)
        beta2: spectral slope above break (nu >> nu_break)
        s: sharpness parameter from Table 2
        
    Returns:
        Local spectral index beta(nu)
    """
    x = np.asarray(nu_ratio, dtype=float)
    
    # Compute flux at x and x * (1 + dx) for numerical derivative
    dx = 1e-6
    f0 = flux_around_break(x, beta1, beta2, s)
    f1 = flux_around_break(x * (1.0 + dx), beta1, beta2, s)
    
    # beta = d(ln F)/d(ln nu)  [NOT negative! F_nu ~ nu^beta]
    d_ln_f = np.log(f1 / f0)
    d_ln_nu = np.log(1.0 + dx)
    
    return d_ln_f / d_ln_nu


def sharpness_parameter_cooling(p):
    """
    Sharpness parameter s for the cooling break (break 3 in Table 2).
    
    From Granot & Sari 2001 Table 2, for the nu_c break:
    |s|(p) = 1.15 - 0.06*p  (fitted for p in [2.2, 2.5, 3.0])
    
    The sign of s equals sign(β1 - β2). For cooling break:
    β1 = (1-p)/2, β2 = -p/2
    β1 - β2 = (1-p)/2 - (-p/2) = (1-p+p)/2 = 1/2 > 0, so s > 0.
    
    Args:
        p: electron power-law index
        
    Returns:
        s: sharpness parameter (positive for cooling break)
    """
    return 1.15 - 0.06 * p


def build_cooling_break_interpolator(p_values=None, nu_ratio_range=(-5, 5), n_points=2000):
    """
    Build 2D interpolator for spectral index around cooling break.
    
    Creates a lookup table: beta(p, log10(nu/nu_c))
    
    Args:
        p_values: array of p values to compute (default: 2.0 to 3.0)
        nu_ratio_range: tuple of (log10_min, log10_max) for nu/nu_c
        n_points: number of grid points in nu/nu_c direction
        
    Returns:
        interpolator: callable with signature beta(p, log10_nu_ratio)
    """
    if p_values is None:
        p_values = np.linspace(2.0, 3.0, 51)
    
    log_nu_ratios = np.linspace(nu_ratio_range[0], nu_ratio_range[1], n_points)
    nu_ratios = 10.0 ** log_nu_ratios
    
    # Grid: (p, log10(nu/nu_c))
    beta_grid = np.zeros((len(p_values), len(log_nu_ratios)))
    
    for i, p in enumerate(p_values):
        # Cooling break from G&S Table 2: beta1 = (1-p)/2, beta2 = -p/2
        # These are negative because synchrotron flux decreases with frequency
        beta1 = (1.0 - p) / 2.0
        beta2 = -p / 2.0
        s = sharpness_parameter_cooling(p)
        
        beta_grid[i, :] = local_spectral_index(nu_ratios, beta1, beta2, s)
    
    return RegularGridInterpolator(
        (p_values, log_nu_ratios),
        beta_grid,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolate beyond bounds
    )


def sharpness_parameter_nu_m(p):
    """
    Sharpness parameter s for the nu_m break (break 2 in Table 2).
    
    From Granot & Sari 2001 Table 2, for the nu_m break:
    |s|(p) = 1.84 - 0.40*p  (ISM, k=0)
    
    The sign of s equals sign(β1 - β2). For nu_m break in slow cooling:
    β1 = 1/3, β2 = (1-p)/2
    β1 - β2 = 1/3 - (1-p)/2 = (2 - 3(1-p))/6 = (3p - 1)/6 > 0 for p > 1/3, so s > 0.
    
    Args:
        p: electron power-law index
        
    Returns:
        s: sharpness parameter (positive for nu_m break)
    """
    return 1.84 - 0.40 * p


def build_nu_m_break_interpolator(p_values=None, nu_ratio_range=(-5, 5), n_points=2000):
    """
    Build 2D interpolator for spectral index around nu_m break.
    
    Creates a lookup table: beta(p, log10(nu/nu_m))
    
    Args:
        p_values: array of p values to compute (default: 2.0 to 3.0)
        nu_ratio_range: tuple of (log10_min, log10_max) for nu/nu_m
        n_points: number of grid points in nu/nu_m direction
        
    Returns:
        interpolator: callable with signature beta(p, log10_nu_ratio)
    """
    if p_values is None:
        p_values = np.linspace(2.0, 3.0, 51)
    
    log_nu_ratios = np.linspace(nu_ratio_range[0], nu_ratio_range[1], n_points)
    nu_ratios = 10.0 ** log_nu_ratios
    
    beta_grid = np.zeros((len(p_values), len(log_nu_ratios)))
    
    for i, p in enumerate(p_values):
        # nu_m break in slow cooling from G&S Table 2: beta1 = 1/3, beta2 = (1-p)/2
        # beta2 is negative for optically thin regime
        beta1 = 1.0 / 3.0
        beta2 = (1.0 - p) / 2.0
        s = sharpness_parameter_nu_m(p)
        
        beta_grid[i, :] = local_spectral_index(nu_ratios, beta1, beta2, s)
    
    return RegularGridInterpolator(
        (p_values, log_nu_ratios),
        beta_grid,
        method='linear',
        bounds_error=False,
        fill_value=None
    )


class SpectralIndexCalculator:
    """
    Fast spectral index calculator using precomputed interpolation tables.
    
    This class builds lookup tables for the spectral index as a function of
    proximity to break frequencies, following Granot & Sari 2001.
    
    Usage:
        calc = SpectralIndexCalculator()
        beta = calc.beta_at_frequency(nu_obs, nu_m, nu_c, p)
    """
    
    def __init__(self):
        """Initialize interpolators for nu_m and nu_c breaks."""
        self.interp_nu_m = build_nu_m_break_interpolator()
        self.interp_nu_c = build_cooling_break_interpolator()
    
    def beta_at_frequency(self, nu_obs, nu_m, nu_c, p):
        """
        Compute spectral index at observed frequency.
        
        Accounts for smooth transitions around break frequencies.
        
        Args:
            nu_obs: observed frequency in Hz (scalar or array)
            nu_m: minimum synchrotron frequency in Hz  
            nu_c: cooling frequency in Hz
            p: electron power-law index
            
        Returns:
            beta: spectral index (scalar or array matching nu_obs)
        """
        nu_obs = np.asarray(nu_obs, dtype=float)
        scalar_input = nu_obs.ndim == 0
        nu_obs = np.atleast_1d(nu_obs)
        
        beta = np.zeros_like(nu_obs)
        
        # Determine regime
        if nu_c > nu_m:
            # Slow cooling: nu_m < nu_c
            # Relevant break for XRT band is likely nu_c
            
            for i, nu in enumerate(nu_obs):
                # Use interpolators, choosing based on which break is closest
                log_ratio_m = np.log10(nu / nu_m)
                log_ratio_c = np.log10(nu / nu_c)
                
                # Determine which regime we're in based on absolute distances
                dist_to_m = abs(log_ratio_m)
                dist_to_c = abs(log_ratio_c)
                
                if log_ratio_m < -4.5:
                    # Very far below nu_m: beta = 1/3 (self-absorption)
                    beta[i] = 1.0 / 3.0
                elif dist_to_m < dist_to_c and log_ratio_m < 4.5:
                    # Closer to nu_m than nu_c, and within range: use nu_m interpolator
                    beta[i] = self.interp_nu_m([[p, log_ratio_m]])[0]
                elif log_ratio_c > 4.5:
                    # Very far above nu_c: beta = -p/2
                    beta[i] = -p / 2.0
                elif abs(log_ratio_c) < 4.5:
                    # Near nu_c (closer than nu_m, or nu_m out of range): use nu_c interpolator
                    beta[i] = self.interp_nu_c([[p, log_ratio_c]])[0]
                else:
                    # Between breaks, far from both: beta = (1-p)/2
                    beta[i] = (1.0 - p) / 2.0
        else:
            # Fast cooling: nu_c < nu_m
            # Most observations are likely above both breaks
            
            for i, nu in enumerate(nu_obs):
                # Fast cooling: use interpolators across wide range
                log_ratio_c = np.log10(nu / nu_c)
                log_ratio_m = np.log10(nu / nu_m)
                
                if log_ratio_c < -4.5:
                    # Very far below nu_c: beta = 1/3
                    beta[i] = 1.0 / 3.0
                elif log_ratio_c < 4.5:
                    # Near nu_c break: use interpolator
                    beta[i] = self.interp_nu_c([[p, log_ratio_c]])[0]
                else:
                    # Above nu_c (also above nu_m in fast cooling): beta = -p/2
                    beta[i] = -p / 2.0
        
        return beta[0] if scalar_input else beta


# Global instance for convenience
_CALCULATOR = None


def get_spectral_index_calculator():
    """Get or create global SpectralIndexCalculator instance."""
    global _CALCULATOR
    if _CALCULATOR is None:
        _CALCULATOR = SpectralIndexCalculator()
    return _CALCULATOR
