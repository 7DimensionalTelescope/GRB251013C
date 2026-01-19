
def power_law_prior():
    return {
        "F0": (1e-15, 1e1),
        "alpha": (0, 5),
    }, {"F0": "log_uniform", "alpha": "uniform"}

def smooth_broken_power_law_prior():
    return {
        "F0": (1e-5, 1e2),
        "alpha": (0, 10),
        "beta": (0, 10),
    }, {"F0": "log_uniform", "alpha": "uniform", "beta": "uniform"}