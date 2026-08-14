"""Priors for the fits.

ParamDefWithPrior wraps VegasAfterglow's ParamDef to add an optional Gaussian
prior on top of the box bounds (used by the afterglow MCMC fit). The
*_prior functions below provide default bounds/priors for the empirical
curve-fit models.
"""
from VegasAfterglow import ParamDef, Scale


class ParamDefWithPrior:
    """Extended ParamDef that supports gaussian priors."""

    def __init__(self, name, lower, upper, scale=Scale.LINEAR, initial=None, gaussian_prior=None):
        self.param_def = ParamDef(name, lower, upper, scale, initial)
        self.gaussian_prior = gaussian_prior

    @property
    def name(self):
        return self.param_def.name

    @property
    def lower(self):
        return self.param_def.lower

    @property
    def upper(self):
        return self.param_def.upper

    @property
    def scale(self):
        return self.param_def.scale

    @property
    def initial(self):
        return self.param_def.initial

    def has_gaussian_prior(self):
        return self.gaussian_prior is not None

    def get_prior_mean_sigma(self):
        if self.gaussian_prior is None:
            return None, None
        return self.gaussian_prior

    def to_param_def(self):
        """Return the underlying ParamDef for VegasAfterglow compatibility."""
        return self.param_def


def power_law_prior(data_type="lightcurve", **kwargs):
    if data_type == "lightcurve":
        default_param_bounds = {
            "F_pl": (1e-15, 1e1),
            "alpha": (0, 5),
        }
        default_prior = {"F_pl": "log_uniform", "alpha": "uniform"}
    elif data_type == "sed":
        default_param_bounds = {
            "F_pl": (1e-2, 1e3),
            "beta": (0, 3),
        }
        default_prior = {"F_pl": "log_uniform", "beta": "uniform"}
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    return update_params(default_param_bounds, default_prior, kwargs)

def broken_power_law_prior(**kwargs):
    default_param_bounds = {
        "F_bpl": (1e-15, 1e1),
        "t_break": (1e-1, 100),
        "alpha_1": (-5, 5),
        "alpha_2": (0, 5),
    }
    default_prior = {"F_bpl": "log_uniform", "t_break": "log_uniform", "alpha_1": "uniform", "alpha_2": "uniform"}
    return update_params(default_param_bounds, default_prior, kwargs)

def smooth_broken_power_law_prior(**kwargs):
    default_param_bounds = {
        "F_sbpl": (1e-15, 1e1),
        "t_break": (2, 100),
        "alpha_r": (0, 5),
        "alpha_d": (0, 5),
        "smooth_power": (1e-2, 1),
    }
    default_prior = {"F_sbpl": "log_uniform", "t_break": "log_uniform", "alpha_r": "uniform", "alpha_d": "uniform", "smooth_power": "log_uniform"}
    return update_params(default_param_bounds, default_prior, kwargs)

def host_galaxy_extinction_prior(**kwargs):
    default_param_bounds = {
        "Av": (-0.82, 0.41),
    }
    default_prior = {"Av": "log_norm"}
    return update_params(default_param_bounds, default_prior, kwargs)

def update_params(default_bounds, default_prior, kwargs):
    if kwargs is not None:
        for key, value in kwargs.items():
            if isinstance(value, tuple) and len(value) == 2:
                default_bounds[key] = value
            elif isinstance(value, str):
                default_prior[key] = value
            else:
                raise ValueError(f"Invalid input in kwargs: key/value={key}/{value}")
                
    return default_bounds, default_prior
