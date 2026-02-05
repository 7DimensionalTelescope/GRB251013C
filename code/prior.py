
def power_law_prior():
    return {
        "F_pl": (1e-15, 1e1),
        "alpha": (0, 5),
    }, {"F_pl": "log_uniform", "alpha": "uniform"}

def broken_power_law_prior():
    return {
        "F_bpl": (1e-15, 1e1),
        "t_break": (1e-1, 100),
        "alpha_1": (0, 5),
        "alpha_2": (0, 5),
    }, {"F_bpl": "log_uniform", "t_break": "log_uniform", "alpha_1": "uniform", "alpha_2": "uniform"}

def smooth_broken_power_law_prior():
    return {
        "F_sbpl": (1e-15, 1e1),
        "t_break": (2, 100),
        "alpha_r": (0, 5),
        "alpha_d": (0, 5),
        "smooth_power": (1e-2, 1),
    }, {"F_sbpl": "log_uniform", "t_break": "log_uniform", "alpha_r": "uniform", "alpha_d": "uniform", "smooth_power": "log_uniform"}