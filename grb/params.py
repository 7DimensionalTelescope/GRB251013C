"""Parameter definitions for the afterglow fit.

ParamDefWithPrior wraps VegasAfterglow's ParamDef to add an optional Gaussian
prior on top of the box bounds. make_param_defs builds the parameter set for
the combined core + reverse-shock + flare + wing model.
"""
from VegasAfterglow import ParamDef, Scale

from .const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA


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


def default_nwalkers(ndim):
    return max(4 * ndim, 32)


def make_param_defs(include_flare=True, include_wing=True):
    """Parameter definitions

    Updated to allow extremely narrow jets (like GRB 221009A: ~0.6-0.8 degrees)
    theta_c_core: 0.001 to 0.052 rad (0.057° to 3°)
    - Lower limit allows ultra-narrow jets
    - Upper limit keeps core reasonably collimated
    """
    params = [
        # Core jet (narrow range to avoid bimodal distribution)
        ParamDefWithPrior("E_iso_core", 5e51, 1e53, Scale.LOG),
        ParamDefWithPrior("Gamma0_core", 300, 1100, Scale.LOG),  # Extended for narrow jets
        ParamDefWithPrior("theta_c_core", 0.001, 0.04, Scale.LOG),  # 0.057° to 1.7° (avoid bimodal)

        # Environment & forward shock microphysics
        ParamDefWithPrior("n_ism", 5, 150, Scale.LOG),  # Extended: high density environment
        #ParamDefWithPrior("p", 2.1, 2.5, Scale.LINEAR),
        ParamDefWithPrior("p", 2.01, 2.3, Scale.LINEAR),
        ParamDefWithPrior("eps_e", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B", 0.005, 0.05, Scale.LOG),
        ParamDefWithPrior("xi", 0.8, 1.0, Scale.LINEAR),
        ParamDefWithPrior("tau", 5, 30, Scale.LOG),  # Tighter: 5-30s to prevent late RS peak

        # Reverse shock (constrained to prevent unphysical late peak)
        ParamDefWithPrior("p_r", 2.0, 3.0, Scale.LINEAR),
        ParamDefWithPrior("eps_e_r", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B_r", 0.005, 0.3, Scale.LOG),  # Lower upper limit: prevent RS dominance
        ParamDefWithPrior("xi_r", 0.7, 1.0, Scale.LINEAR),

        # Host extinction
        ParamDefWithPrior(
            "A_V", 0.001, 2.0, Scale.LOG,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]

    if include_flare:
        params.extend([
            ParamDefWithPrior("t_start_flare", 1000, 5000, Scale.LOG),  # Wider range
            ParamDefWithPrior("tau_rise_flare", 30, 2000, Scale.LOG),  # Lower limit for fast rise
            ParamDefWithPrior("tau_decay_flare", 1000, 10000, Scale.LOG),  # Extended
            ParamDefWithPrior("A_flare", 1e-10, 5e-9, Scale.LOG),  # Extended: allow brighter flares
            ParamDefWithPrior("flare_beta", 0.5, 1.2, Scale.LINEAR),
        ])

    if include_wing:
        params.extend([
            ParamDefWithPrior("E_iso_wing", 1e52, 1e53, Scale.LOG),  # Wider range
            ParamDefWithPrior("Gamma0_wing", 10, 100, Scale.LOG),  # Extended upper limit
            ParamDefWithPrior("theta_c_wing", 0.2, 0.5, Scale.LOG),  # Wider wing: 11-29° (helps late-time emission)
            ParamDefWithPrior("p_wing", 2.2, 2.9, Scale.LINEAR),  # Extended: allow steeper spectrum
            ParamDefWithPrior("eps_e_wing", 0.3, 1.0, Scale.LOG),
            ParamDefWithPrior("eps_B_wing", 0.001, 0.02, Scale.LOG),
            ParamDefWithPrior("xi_wing", 0.6, 1.0, Scale.LINEAR),
        ])

    return params
