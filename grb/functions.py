import numpy as np

def pivot_finder(x, yerr):
    weights = 1.0 / (yerr**2)
    log_pivot = np.average(np.log(x), weights=weights)
    x_pivot = np.exp(log_pivot)
    return x_pivot

def power_law(x, F0, index, x0 = 1):
    return F0 * np.power(x / x0, -index)

def power_law_log(x, log_F0, index, x0 = 1):
    return np.log10(power_law(x, 10**log_F0, index, x0))
                                                     
def broken_power_law(t, F_break, t_break, alpha_1, alpha_2, t0 = 1):
    """
    F(t) = F_break * (t / tbreak)^(-alpha1) for t < t_break
    F(t) = F_break * (t / tbreak)^(-alpha2) for t >= t_break
    """
    t_safe = np.maximum(t, 1e-10)
    t0_safe = max(t0, 1e-10)
    t_ratio = t_safe / t0_safe
    result = np.where(t_safe < t_break,
                      F_break * (t / t_break)**(-alpha_1),
                      F_break * (t / t_break)**(-alpha_2))
    return np.maximum(result, 1e-100)

def broken_power_law_log(t, log_F_break, log_t_break, alpha1, alpha2, t0 = 1):
    """
    F(t) = F_break * (t / t0)^(-alpha1) for t < t_break
    F(t) = F_break * t_break^(alpha2-alpha1) * (t / t0)^(-alpha2) for t >= t_break
    """
    F = broken_power_law(t, np.power(10, log_F_break), np.power(10, log_t_break), alpha1, alpha2)
    return np.log10(F)


def smooth_broken_power_law(t, F0, tb, alpha_r, alpha_d, smooth_power):
    """
    F(t) = F0 * (t/tb)^(-alpha_r) * (0.5 * (1 + (t/tb)^(1/smooth_power)))**((alpha_r - alpha_d) * smooth_power)
    """

    ratio = t / tb
    
    # Power law component
    term1 = ratio**(-alpha_r)
    
    
    smooth_factor = (0.5 * (1 + ratio**(1/smooth_power)))**((alpha_r - alpha_d) * smooth_power)
    
    return F0 * term1 * smooth_factor
    

def smooth_broken_power_law_log(t, log_F0, log_tb, alpha_r, alpha_d, kappa):
    """
    F(t) = F0 / [(t/tb)^(kappa*alpha_r) + (t/tb)^(kappa*alpha_d)]^(1/kappa)
    """
    F = smooth_broken_power_law(t, 10**log_F0, 10**log_tb, alpha_r, alpha_d, kappa)
    return np.log10(F)


def two_component_model(t, log_F_pl, alpha,
                        log_F0_smooth, log_tb_smooth, alpha_r, alpha_d, kappa):
    """
    F(t) = F_broken(t) + F_smooth(t)
    """
    F_broken = 10**power_law_log(t, log_F_pl, alpha) 
    F_smooth = 10**smooth_broken_power_law_log(t, log_F0_smooth, log_tb_smooth, alpha_r, alpha_d, kappa)
    return F_broken + F_smooth


def norris_flare(t, t_start, tau_rise, tau_decay, amplitude):
    """Norris function for GRB flare profile"""
    flux = np.zeros_like(t, dtype=float)
    mask = t > t_start
    
    if np.any(mask):
        dt = t[mask] - t_start
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            flux[mask] = amplitude * np.exp(-tau_rise / dt - dt / tau_decay)
        flux[~np.isfinite(flux)] = 0.0
    
    return flux

