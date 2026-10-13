"""Shared helpers for mixing_search.py and sweep_search.py.

Both scripts explore the afterglow parameter space before/around MCMC:
mixing_search.py draws a joint scrambled Sobol sequence over the whole prior
box, sweep_search.py varies one parameter at a time around a baseline. This
module holds only what's genuinely used by both -- data/parameter loading,
the coupled physical-validity rules, sampled<->physical conversions, and the
multiprocessing worker context. No matplotlib/pandas/scipy here; those are
mode-specific.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import multiprocessing as mp
import os

# Keep each worker single-threaded so pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

os.sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fitting"))

import numpy as np

from VegasAfterglow import Scale

from grb.params import ParamDefWithPrior

from grb.functions import FLARE_SHAPES
from grb.modeling import load_all_optical_data
from grb.params import ENVIRONMENTS, make_param_defs
from grb.plotting import C_AA_PER_S
from grb.spectral_index import load_xrt_spectral_index

DEFAULT_RESULTS_ROOT = Path("/home/hongyp007/yp_folder/projects/GRB/result")

STATUS_SUCCESS = 0
STATUS_PHYSICAL_INVALID = 1
STATUS_NUMERICAL_FAILURE = 2
STATUS_NAMES = {
    STATUS_SUCCESS: "success",
    STATUS_PHYSICAL_INVALID: "physical_invalid",
    STATUS_NUMERICAL_FAILURE: "numerical_failure",
}

PHYSICAL_VALIDITY_RULES = [
    "eps_e + eps_B <= 1 in each shock region (FS, RS, wing)",
    "theta_c_wing > theta_c_core (only when the wing is included)",
    "tau_decay_flare > tau_rise_flare (only when the flare is included)",
]


def add_common_args(parser, outdir_help):
    """Flags identical between mixing_search.py and sweep_search.py.

    outdir_help is passed in since the default-directory-name prose differs
    per script (mixing_<environment>_<timestamp> vs sweep_<environment>_<timestamp>).
    """
    parser.add_argument("--ncpus", type=int, default=20)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="ism",
                        help="Circumburst medium: ism (n_ism), wind (A_star), "
                             "or stratified (A_star + n_ism + n0, k_m fixed at 2)")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--flare-function", dest="flare_shape", choices=FLARE_SHAPES, default="norris",
                        help="Flare temporal profile shape when --include-flare (default: norris).")
    parser.add_argument("--include-wing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-cal", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cal-uncert", type=float, default=0.1)
    parser.add_argument("--free-xi-e", default=False, action=argparse.BooleanOptionalAction,
                        help="Add the forward-shock xi_e (0.01-1, linear) as a parameter "
                             "(otherwise fixed at 1)")
    parser.add_argument("--free-xi-e-r", default=False, action=argparse.BooleanOptionalAction,
                        help="Add the reverse-shock xi_e_r (0.01-1, linear) as a parameter "
                             "(otherwise fixed at 1)")
    parser.add_argument("--use-spectral-index", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--outdir", default=None, help=outdir_help)


def sampled_bounds(param_defs):
    """Bounds in the sampled (emcee) coordinates: log10 for LOG parameters."""
    lowers, uppers = [], []
    for p in param_defs:
        if p.scale is Scale.LOG:
            lowers.append(np.log10(p.lower))
            uppers.append(np.log10(p.upper))
        else:
            lowers.append(p.lower)
            uppers.append(p.upper)
    return np.array(lowers), np.array(uppers)


def theta_to_params(theta, param_defs):
    """Sampled vector -> physical-parameter dict (same convention as log_likelihood)."""
    params = {}
    for p, value in zip(param_defs, theta):
        params[p.name] = 10 ** value if p.scale is Scale.LOG else float(value)
    return params


def params_to_theta(params, param_defs):
    """Physical-parameter dict -> sampled vector (inverse of theta_to_params)."""
    return np.array([np.log10(params[p.name]) if p.scale is Scale.LOG else float(params[p.name])
                     for p in param_defs])


def check_physical_validity(thetas, param_defs):
    """Coupled validity rules on top of the box bounds, vectorized over samples.

    Returns an object array of rejection reasons, '' where valid. Rule
    precedence: eps sums (FS, RS, wing), then jet geometry, then flare shape.
    """
    cols = {p.name: j for j, p in enumerate(param_defs)}

    def col(name):
        if name not in cols:
            return None
        p = param_defs[cols[name]]
        values = thetas[:, cols[name]]
        return 10 ** values if p.scale is Scale.LOG else values

    reasons = np.full(len(thetas), "", dtype=object)
    for suffix in ("", "_r", "_wing"):
        eps_e, eps_B = col("eps_e" + suffix), col("eps_B" + suffix)
        if eps_e is not None and eps_B is not None:
            bad = (eps_e + eps_B > 1.0) & (reasons == "")
            reasons[bad] = f"eps_e{suffix}+eps_B{suffix}>1"
    theta_wing, theta_core = col("theta_c_wing"), col("theta_c_core")
    if theta_wing is not None:
        bad = (theta_wing <= theta_core) & (reasons == "")
        reasons[bad] = "theta_c_wing<=theta_c_core"
    tau_decay, tau_rise = col("tau_decay_flare"), col("tau_rise_flare")
    if tau_decay is not None and tau_rise is not None:
        bad = (tau_decay <= tau_rise) & (reasons == "")
        reasons[bad] = "tau_decay_flare<=tau_rise_flare"
    return reasons


# --- worker context (sent once per worker via the Pool initializer) ---
_CTX = {}


def _init_worker(param_defs, xrt_data, optical_datasets, include_flare, include_wing,
                 xrt_index_data, flare_shape, t_grid, lc_bands, sed_time=None, nu_grid=None):
    _CTX["param_defs"] = param_defs
    _CTX["xrt_data"] = xrt_data
    _CTX["optical_datasets"] = optical_datasets
    _CTX["include_flare"] = include_flare
    _CTX["include_wing"] = include_wing
    _CTX["xrt_index_data"] = xrt_index_data
    _CTX["flare_shape"] = flare_shape
    _CTX["t_grid"] = t_grid
    _CTX["lc_bands"] = lc_bands
    _CTX["lc_data"] = [d for d in optical_datasets if not d["name"].startswith("7DT_")]
    _CTX["sed_time"] = sed_time
    _CTX["nu_grid"] = nu_grid


def make_progress(total):
    try:
        from tqdm import tqdm
        return tqdm(total=total, unit="sample")
    except ImportError:
        return None


def _param_defs_json(param_defs):
    """Per-parameter bounds/scale/prior block shared by both run configs."""
    lowers, uppers = sampled_bounds(param_defs)
    return [
        {
            "name": p.name,
            "lower": p.lower,
            "upper": p.upper,
            "scale": "log" if p.scale is Scale.LOG else "linear",
            "sampled_lower": lo,
            "sampled_upper": hi,
            "gaussian_prior": list(p.gaussian_prior) if p.has_gaussian_prior() else None,
        }
        for p, lo, hi in zip(param_defs, lowers, uppers)
    ]


def load_run_inputs(args):
    """Data + parameter setup shared by the mixing and sweep modes."""
    print("Loading ALL data (XRT + all optical bands)...")
    xrt_data, optical_datasets = load_all_optical_data()

    xrt_index_data = None
    if args.use_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
        except Exception as e:
            print(f"  Warning: could not load XRT spectral index ({e}); continuing without it")

    param_defs = make_param_defs(
        include_flare=args.include_flare,
        include_wing=args.include_wing,
        include_cal=args.include_cal,
        cal_uncert=args.cal_uncert,
        environment=args.environment,
        flare_shape=args.flare_shape,
    )
    # Same slots/bounds as modeling.early_phase_model.make_param_defs: xi_e before tau, xi_e_r before A_V
    if args.free_xi_e:
        pos = next(i for i, p in enumerate(param_defs) if p.name == "tau")
        param_defs.insert(pos, ParamDefWithPrior("xi_e", 0.01, 1.0, Scale.LINEAR))
    if args.free_xi_e_r:
        pos = next(i for i, p in enumerate(param_defs) if p.name == "A_V")
        param_defs.insert(pos, ParamDefWithPrior("xi_e_r", 0.01, 1.0, Scale.LINEAR))
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    param_names = [p.name for p in param_defs]

    n_cpus = mp.cpu_count()
    n_workers = max(1, min(args.ncpus, n_cpus - 2))  # leave headroom on the shared machine

    # Display time grid for plotted models (same construction as plot_light_curves)
    lc_data = [d for d in optical_datasets if not d["name"].startswith("7DT_")]
    lc_bands = [(d["name"], d["frequency"]) for d in lc_data]
    t_min = min(xrt_data["time"].min(), min(d["time"].min() for d in lc_data))
    t_max = max(xrt_data["time"].max(), max(d["time"].max() for d in lc_data))
    t_grid = np.geomspace(t_min * 0.8, t_max * 1.2, 300)

    # 7DT spectrum grid: wavelength range from the 7DT data, evaluated at its median
    # observation time (same convention as grb.plotting.render_light_curve_figure)
    sdt_data = [d for d in optical_datasets if d["name"].startswith("7DT_")]
    sed_time = float(np.median([d["time"][0] for d in sdt_data])) if sdt_data else None
    if sdt_data:
        sdt_nu = np.array([d["frequency"] for d in sdt_data])
        sdt_wavelength = C_AA_PER_S / sdt_nu
        wavelength_grid = np.geomspace(sdt_wavelength.min() * 0.9, sdt_wavelength.max() * 1.1, 300)
        nu_grid = C_AA_PER_S / wavelength_grid
    else:
        wavelength_grid = nu_grid = None

    return SimpleNamespace(
        xrt_data=xrt_data, optical_datasets=optical_datasets, xrt_index_data=xrt_index_data,
        param_defs=param_defs, labels=labels, param_names=param_names,
        n_cpus=n_cpus, n_workers=n_workers, lc_data=lc_data, lc_bands=lc_bands, t_grid=t_grid,
        sdt_data=sdt_data, sed_time=sed_time, wavelength_grid=wavelength_grid, nu_grid=nu_grid)
