
def power_law_prior():
    return {
        "F0": (1e-15, 1e1),
        "alpha": (0, 5),
    }, {"F0": "log_uniform", "alpha": "uniform"}

def broken_power_law_prior():
    return {
        "F_break": (1e-15, 1e1),
        "t_break": (1e-2, 1e3),
        "alpha1": (0, 5),
        "alpha2": (0, 5),
    }, {"F_break": "log_uniform", "t_break": "log_uniform", "alpha1": "uniform", "alpha2": "uniform"}
