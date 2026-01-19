import numpy as np

def power_law(t, F0, alpha, t0 = 1):
    return F0 * np.power(t / t0, -alpha)

def power_law_log(t, log_F0, alpha, t0 = 1):
    return np.log10(power_law(t, 10**log_F0, alpha, t0))
                                                     
def broken_power_law(t, F_break, t_break, alpha1, alpha2):
    """
    F(t) = F_break * (t / t_break)^(-alpha1) for t < t_break
    F(t) = F_break * (t / t_break)^(-alpha2) for t >= t_break
    """
    t_safe = np.maximum(t, 1e-10)
    t_break_safe = max(t_break, 1e-10)
    
    t_ratio = t_safe / t_break_safe
    
    result = np.where(t_safe < t_break_safe,
                     F_break * t_ratio**(-alpha1),  # Early: t < t_break
                     F_break * t_ratio**(-alpha2))  # Late: t ≥ t_break
    
    return np.maximum(result, 1e-100)  # Ensure positive


def broken_power_law_log(t, log_F_break, log_t_break, alpha1, alpha2):
    """
    F(t) = F_break * (t / t_break)^(-alpha1) for t < t_break
    F(t) = F_break * (t / t_break)^(-alpha2) for t >= t_break
    """
    F = broken_power_law(t, np.power(10, log_F_break), np.power(10, log_t_break), alpha1, alpha2)
    return np.log10(F)


def smooth_broken_power_law(t, F0, tb, alpha_r, alpha_d, kappa):
    """
    F(t) = F0 / [(t/tb)^(kappa*alpha_r) + (t/tb)^(kappa*alpha_d)]^(1/kappa)
    """
    t_safe = np.maximum(t, 1e-10)
    tb_safe = max(tb, 1e-10)
    
    ratio = t_safe / tb_safe
    term1 = np.power(ratio, kappa * alpha_r)
    term2 = np.power(ratio, kappa * alpha_d)
    denominator = np.power(term1 + term2, 1.0 / kappa)
    
    result = F0 / np.maximum(denominator, 1e-100)
    return np.maximum(result, 1e-100)


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
