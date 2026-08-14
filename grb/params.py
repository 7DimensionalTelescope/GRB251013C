"""Parameter definitions for the afterglow fit.

make_param_defs builds the parameter set for the combined
core + reverse-shock + flare + wing model.
"""
from VegasAfterglow import Scale

from .const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA
from .prior import ParamDefWithPrior


def default_nwalkers(ndim):
    return max(4 * ndim, 32)


# Optical LC datasets that are *not* the 7DT flux reference. Each gets an
# optional multiplicative calibration scale when include_cal=True.
NON_7DT_CAL_DATASETS = ("i-band", "Leavitt_Rc", "Leavitt_Ic")


def cal_param_name(dataset_name):
    """Map a dataset name (e.g. 'i-band') to its cal parameter (cal_i_band)."""
    return "cal_" + dataset_name.replace("-", "_")


def dataset_cal_factor(params, dataset_name):
    """Multiplicative cal scale for a dataset. 7DT (and missing keys) -> 1."""
    if dataset_name.startswith("7DT_"):
        return 1.0
    return float(params.get(cal_param_name(dataset_name), 1.0))


def make_param_defs(include_flare=True, include_wing=True, include_cal=False, cal_uncert=0.1):
    """Parameter definitions

    Bounds retuned (2026-07-30) from the joint re-optimization of the
    final_flare_wing_20260724_171919 best fit inside a widened box
    (logL -577.6 -> -548.0, total chi2 1154.9 -> 1095.8 on 232 points):
    - tau_rise_flare and p_wing: the improved optimum sits OUTSIDE the old
      bounds (25.5 s < 30 s; 3.06 > 2.9), so widening these is required.
    - theta_c_core, n_ism, eps_B, E_iso_wing, eps_e_wing, theta_c_wing:
      posterior modes hug the old walls; widened so the posterior can close.
    - p and the eps_B lower range are deliberately NOT opened further:
      chasing the observed XRT photon index (~1.88, vs model floor ~2.04)
      via low eps_B / high p was tested and loses badly (dlogL <= -620) -
      the spectral-index tension (chi2 ~ 88 for 45 pts) is a model
      limitation, not a bounds artifact.

    include_cal: add one LINEAR multiplicative scale per non-7DT optical LC
    dataset (i-band, Leavitt_Rc, Leavitt_Ic), bounded to
    [1 - cal_uncert, 1 + cal_uncert]. 7DT is the flux reference (scale=1).
    """
    params = [
        # Core jet (narrow range to avoid bimodal distribution)
        ParamDefWithPrior("E_iso_core", 5e51, 1e53, Scale.LOG),
        ParamDefWithPrior("Gamma0_core", 300, 1100, Scale.LOG), 
        #ParamDefWithPrior("theta_c_core", 0.001, 0.08, Scale.LOG), 
        ParamDefWithPrior("theta_c_core", 0.001, 1.0, Scale.LOG), 

        # Environment & forward shock microphysics
        ParamDefWithPrior("n_ism", 5, 400, Scale.LOG),
        #ParamDefWithPrior("p", 2.1, 2.5, Scale.LINEAR),
        ParamDefWithPrior("p", 2.01, 2.3, Scale.LINEAR),
        ParamDefWithPrior("eps_e", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B", 0.002, 0.05, Scale.LOG), 
        ParamDefWithPrior("xi", 0.8, 1.0, Scale.LINEAR),
        ParamDefWithPrior("tau", 5, 30, Scale.LOG),  

        # Reverse shock (constrained to prevent unphysical late peak)
        ParamDefWithPrior("p_r", 2.0, 3.0, Scale.LINEAR),
        ParamDefWithPrior("eps_e_r", 0.02, 0.1, Scale.LOG),
        ParamDefWithPrior("eps_B_r", 0.005, 0.3, Scale.LOG),  
        ParamDefWithPrior("xi_r", 0.7, 1.0, Scale.LINEAR),

        # Host extinction
        ParamDefWithPrior(
            "A_V", 0.001, 2.0, Scale.LOG,
            gaussian_prior=(HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA),
        ),
    ]

    if include_flare:
        params.extend([
            ParamDefWithPrior("t_start_flare", 1000, 5000, Scale.LOG), 
            ParamDefWithPrior("tau_rise_flare", 10, 2000, Scale.LOG), 
            ParamDefWithPrior("tau_decay_flare", 1000, 10000, Scale.LOG),
            ParamDefWithPrior("A_flare", 1e-10, 5e-9, Scale.LOG),  
            ParamDefWithPrior("flare_beta", 0.5, 1.2, Scale.LINEAR),
        ])

    if include_wing:
        params.extend([
            ParamDefWithPrior("E_iso_wing", 1e51, 1e53, Scale.LOG), 
            ParamDefWithPrior("Gamma0_wing", 10, 100, Scale.LOG),
            ParamDefWithPrior("theta_c_wing", 0.2, 0.7, Scale.LOG),
            ParamDefWithPrior("p_wing", 2.2, 2.9, Scale.LINEAR),  
            ParamDefWithPrior("eps_e_wing", 0.1, 1.0, Scale.LOG),  
            ParamDefWithPrior("eps_B_wing", 0.001, 0.02, Scale.LOG),
            ParamDefWithPrior("xi_wing", 0.6, 1.0, Scale.LINEAR),
        ])

    if include_cal:
        if not (0.0 < cal_uncert < 1.0):
            raise ValueError(f"cal_uncert must be in (0, 1), got {cal_uncert}")
        lo, hi = 1.0 - cal_uncert, 1.0 + cal_uncert
        for name in NON_7DT_CAL_DATASETS:
            params.append(ParamDefWithPrior(cal_param_name(name), lo, hi, Scale.LINEAR))

    return params
