#%%
from dataclasses import dataclass
import emcee
import corner
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from VegasAfterglow import logscale_screen, Observer, Wind, ISM, TophatJet, Radiation, Model, PowerLawWing, Magnetar
from VegasAfterglow.units import keV, mJy, hr, sec, Hz, _c_A
from grb.const import REDSHIFT, RA, DEC, TRIGGER_TIME, AV, FILTER_INFO
from grb.io import read_data, filter_data
from astropy.cosmology import FlatLambdaCDM
import warnings
from scipy.optimize import differential_evolution
import dynesty
from dynesty import utils as dyfunc
from scipy.special import logsumexp

warnings.filterwarnings("ignore", category=FutureWarning, message="ChainedAssignmentError")

cosmo = FlatLambdaCDM(H0=67.66, Om0=0.27) # Use same value from VegasAfterglow
LUMI_DIST = cosmo.luminosity_distance(REDSHIFT).to('cm').value

_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
IMAGE_DIR = "/home/hongyp007/hongyp/GRB251013C/GRB_modelling/lcgen_result"
RE_DIR = None
RE_DIR ="/home/hongyp007/hongyp/GRB251013C/GRB_modelling/lcgen_result/stratified/20260630_210046"
SKIP_CORNER = True
use_i_band = True
ONLY_PLOT = True
PLOT_DIR = "/home/hongyp007/hongyp/GRB251013C/GRB_modelling"

BAND_STYLE = {
    "XRT":   ("black",       "X-ray"),
    "g":     ("dodgerblue",  "g"),
    "V":     ("limegreen",   "V"),
    "r":     ("gold",        "r"),
    "Rc":    ("darkorange",  "Rc"),
    "i":     ("red",         "i"),
    "Ic":    ("darkred",     "Ic"),
    "J":     ("deeppink",    "J"),
    "clear": ("slategray",   "clear"),
}

LC_T_RANGE   = np.logspace(2, 6, 200)   # 100 s → 10^6 s
SPEC_F_RANGE = np.logspace(14, 15, 200) # 10^14 Hz → 10^15 Hz

#%%
####################### XRT #######################
xrt_data = read_data("xrt")
# xrt_data = read_data("xrt_unabsorb")

XRT_EXCLUDE_TIME_RANGE = (3e3, 1e4)  # Flare, excluded from fit but still shown on the LC plot
filtered_xrt_data = filter_data(xrt_data, exclude_time_range=XRT_EXCLUDE_TIME_RANGE) # Exclude flare

xrt_err = np.array([max(np.abs(high), np.abs(low)) for high, low
                    in zip(filtered_xrt_data["flux_high"], filtered_xrt_data["flux_low"])])

# Full (unfiltered) data and its errors, for plotting only — the fit uses filtered_xrt_data
xrt_err_all = np.array([max(np.abs(high), np.abs(low)) for high, low
                    in zip(xrt_data["flux_high"], xrt_data["flux_low"])])

xray_array = np.column_stack([
    filtered_xrt_data['time'].to_numpy(),
    filtered_xrt_data['flux'].to_numpy(),
    xrt_err
])  # shape (N, 3): [time, flux, err]
xray_data = [xray_array]
XRT_NU_MIN = 0.3 * keV  # 0.3 keV in Hz
XRT_NU_MAX = 10 * keV  # 10.0 keV in Hz
print(f"Added {len(xray_data[0])} points for XRT")

####################### 7DT SED #######################
sdt_data = read_data("sdt_pivot", correct_galactic_extinction=True, correct_host_extinction=True, 
                     host_av=AV, host_z=REDSHIFT, add_converted_flux=True)

# Exclude g, r, i band data
sdt_data = sdt_data[~sdt_data["filter"].isin(["g", "r", "i"])]

sdt_time = sdt_data["date_obs"].iloc[0]
sdt_from_t0 = datetime.strptime(sdt_time, '%Y-%m-%dT%H:%M:%S.%f') - TRIGGER_TIME
SDT_SECONDS = sdt_from_t0.total_seconds()

sdt_processed = []
sed_sys_fraction = 0.074   # medium-band cross-calibration floor (from SED self-consistency test)
for _, row in sdt_data.iterrows():
    flux_mJy = row["flux_mJy"] * mJy
    err_mJy  = row["flux_error_mJy"] * mJy
    err_mJy  = np.sqrt(err_mJy**2 + (flux_mJy * sed_sys_fraction)**2)
    sdt_processed.append((
        row["frequency_Hz"],
        np.array([SDT_SECONDS]),
        np.array([flux_mJy]),
        np.array([err_mJy])
    ))

print(f"Added {len(sdt_data)} points for 7DT SED")

####################### GCN Circular #######################
circular_processed = []
circular_plot_data = []

circular_data = read_data("circular_wavelength", correct_galactic_extinction=True, correct_host_extinction=True,
                            host_av=AV, host_z=REDSHIFT, add_converted_flux=True)

# Filter configurations
filter_configs = [
    # {"filter": "g", "facility": "7DT"},
    # {"filter": "r", "facility": "7DT"},
    # {"filter": "i", "facility": "7DT"},
    # {"filter": "i", "facility": "NUTTelA-TAO"},
    # {"filter": "Ic", "facility": "Leavitt"},
    # {"filter": "Rc", "facility": "Leavitt"},
    # {"filter": "V", "facility": "OsservatorioAstronomicoNastroVerde"},
    # {"filter": "J", "facility": "SYSU"},
    # {"filter": "clear", "facility": "Calapai"},
    {"filter":"Rc", "facility":None},
    # {"filter":"R", "facility":None},
    {"filter":"r", "facility":None},
    # {"filter":"i", "facility":None},
]

num_circular = 0
for config in filter_configs:
    df = filter_data(
        circular_data, 
        filter_name=config["filter"], 
        facility_name=config["facility"], 
        remove_upper_limits=True, 
        exclude_time_range=(60, float('inf'))
    )
    
    df = filter_data(df, exclude_time_range=(0, 0.07)) # Exclude early time data

    if df.empty:
        print(f"No data for {config['filter']} from {config['facility']}, skipping...")
        continue
    
    flux = df["flux_mJy"].to_numpy() * mJy
    err = df["flux_error_mJy"].to_numpy() * mJy
    t = df["time"].to_numpy() * hr

    # Add 5% systematic error for cross-calibration uncertainties
    sys_err_fraction = 0.05
    err = np.sqrt(err**2 + (flux * sys_err_fraction)**2)

    # Broadband flux in erg/cm²/s for plotting
    flux_erg = df["flux_erg_cm2_s"].to_numpy()
    err_erg = np.sqrt(df["flux_error_erg_cm2_s"].to_numpy()**2 + (flux_erg * sys_err_fraction)**2)

    # For Rc filter, subsampling the data to avoid too many points
    if config["filter"] == "Rc":
        indices = logscale_screen(t, data_density=20)
        t        = t[indices]
        flux     = flux[indices]
        err      = err[indices]
        flux_erg = flux_erg[indices]
        err_erg  = err_erg[indices]

    nu_val = df["frequency_Hz"].to_numpy()[0]
    nu_bw  = df["frequency_error_Hz"].to_numpy()[0]   # half-bandwidth
    circular_processed.append((config["filter"], nu_val, nu_bw, t, flux, err))
    circular_plot_data.append((config["filter"], t / hr, flux_erg, err_erg))
    
    print(f"Added {len(t)} points for {config['filter']} filter (facility: {config['facility']})")
    num_circular += len(t)

##################### i-band only #####################
if use_i_band:
    iband_processed = []
    iband_plot_data = []
    
    i_data = read_data("i_band", correct_galactic_extinction=True, correct_host_extinction=True, 
                       host_av=AV, host_z=REDSHIFT, add_converted_flux=True)
    i_band_freq = i_data["frequency_Hz"].to_numpy()[0]
    i_band_freq_err = i_data["frequency_error_Hz"].to_numpy()[0]
    
    iband_processed.append((i_data["filter"].to_numpy()[0], 
                            i_band_freq, i_band_freq_err, 
                            i_data["time"].to_numpy() * hr, 
                            i_data["flux_mJy"].to_numpy() * mJy, 
                            i_data["flux_error_mJy"].to_numpy() * mJy))
    iband_plot_data.append((i_data["filter"].to_numpy()[0], 
                            i_data["time"].to_numpy() / hr, 
                            i_data["flux_erg_cm2_s"].to_numpy(), 
                            i_data["flux_error_erg_cm2_s"].to_numpy()))
    
    print(f"Added {len(i_data)} points for i-band filter")

#%%
@dataclass
class Param:
    name: str
    lower: float
    upper: float
    log10: bool = True

# Single source of truth: name, lower bound, upper bound, sampled in log10?
PARAMS = [
    # Medium
    Param("A_star",   -3,    2,    True),  # Wind parameter
    # Param("k_m",      0,    3,    False), # Wind density power-law index (default 2, i.e. n ∝ r^{-2})
    Param("n0",        0,    5,    True),  # ISM density (inner) in cm^-3
    # Param("n_ism",    -5,  0,    True),  # ISM density (outer) in cm^-3
    # Power-law structure parameters
    # Param("k_e",      1,    5,    False),  # Energy power-law index, default 2.0 if not specified
    # Param("k_g",      0.5,    3,    False),  # Lorentz factor power-law, default 2.0 if not specified
    # Core Component
    Param("E_iso",    50,   57,    True),  # Jet energy in erg
    Param("Gamma0",    1,    3,    True),  # Jet initial Lorentz factor
    Param("theta_c",  -3,   -1,    True),  # Jet core opening angle in radians
    Param("eps_e",    -3,  -0.3,   True),  # Electron energy fraction
    Param("eps_B",    -6,  -0.3,   True),  # Magnetic field energy fraction
    Param("p",         2,    3,    False), # Electron spectral index
    Param("xi",       -3,    2,    True),  # Fraction of accelerated electrons
    # Wide Component
    Param("E_iso_w",  50,   57,    True),  # Jet energy in erg
    Param("Gamma0_w",  1,    3,    True),  # Jet initial Lorentz factor
    Param("theta_w",  -3,   -1,    True),  # Jet wing opening angle in radians
    Param("eps_e_w",  -3,  -0.3,   True),  # Electron energy fraction
    Param("eps_B_w",  -6,  -0.3,   True),  # Magnetic field energy fraction
    Param("p_w",       2,    3,    False), # Electron spectral index
    Param("xi_w",     -3,    2,    True),  # Fraction of accelerated electrons
    # Magnetar
    # Param("L0",  44,  48,  True),   # magnetar luminosity at t0 [log10 erg/s]
    # Param("t0",   1,   4,  True),   # spin-down timescale [log10 s]
    # Param("q",    1,   6,  False),  # spin-down power-law index
]

labels = [p.name for p in PARAMS]
pl = np.array([p.lower for p in PARAMS])
pu = np.array([p.upper for p in PARAMS])

import os
import re as _re

def _load_params_from_txt(dirpath):
    """Parse Parameter Setup from bestfit_params.txt in dirpath to reconstruct PARAMS."""
    txt_path = os.path.join(dirpath, "bestfit_params.txt")
    if not os.path.exists(txt_path):
        return None
    pattern = _re.compile(r'Param\("(\w+)",\s*([\d.-]+),\s*([\d.-]+),\s*(log10|linear)\)')
    params, in_setup = [], False
    with open(txt_path) as fh:
        for line in fh:
            if "=== Parameter Setup ===" in line:
                in_setup = True
                continue
            if in_setup:
                if line.startswith("==="):
                    break
                m = pattern.search(line)
                if m:
                    name, lower, upper, scale = m.groups()
                    params.append(Param(name, float(lower), float(upper), scale == "log10"))
    return params or None

_param_names = {p.name for p in PARAMS}
if "A_star" in _param_names and ("n0" in _param_names or "n_ism" in _param_names):
    _MEDIUM_TAG = "stratified"
elif "A_star" in _param_names:
    _MEDIUM_TAG = "wind"
else:
    _MEDIUM_TAG = "ISM"
RUN_DIR = os.path.join(IMAGE_DIR, _MEDIUM_TAG, _RUN_TS, "")

if RE_DIR is None:
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Output directory: {RUN_DIR}")

def to_physical(par):
    return {p.name: 10**v if p.log10 else v for p, v in zip(PARAMS, par)}

def get_model(E_iso, Gamma0, theta_c, theta_w,
              eps_e, eps_B, p,
              E_iso_w, Gamma0_w,
              k_e=2.0, k_g=2.0, p_w=None,
              xi=1.0, xi_w=1.0, eps_e_w=None, eps_B_w=None,
              A_star=0, k_m=2,
              n0=float('inf'), n_ism=0,
              L0=None, t0=None, q=None):

    # Wing inherits the core microphysics / spectral index when not provided
    if eps_e_w is None:
        eps_e_w = eps_e
    if eps_B_w is None:
        eps_B_w = eps_B
    if p_w is None:
        p_w = p
        
    obs = Observer(lumi_dist=LUMI_DIST, z=REDSHIFT, theta_obs=0)
    
    _param_names = {p.name for p in PARAMS}
    if "A_star" in _param_names:
        medium = Wind(A_star=A_star, n0=n0, n_ism=n_ism, k_m=k_m)
    elif "n_ism" in _param_names:
        medium = ISM(n_ism=n_ism)
    else:
        raise ValueError("No medium specified")

    # Central-engine energy injection applied to the core jet when provided
    if L0 is not None:
        jet = TophatJet(E_iso=E_iso, Gamma0=Gamma0, theta_c=theta_c,
                        magnetar=Magnetar(L0=L0, t0=t0, q=q))
    else:
        jet = TophatJet(E_iso=E_iso, Gamma0=Gamma0, theta_c=theta_c)
    rad = Radiation(eps_e=eps_e, eps_B=eps_B, p=p, xi_e=xi, ssc=True, kn=False)
    model_core = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad, resolutions=(0.1, 0.2, 10))

    wing = PowerLawWing(theta_c=theta_w, E_iso_w=E_iso_w, Gamma0_w=Gamma0_w, k_e=k_e, k_g=k_g)
    rad_w = Radiation(eps_e=eps_e_w, eps_B=eps_B_w, p=p_w, xi_e=xi_w)
    model_wing = Model(jet=wing, medium=medium, observer=obs, fwd_rad=rad_w, resolutions=(0.1, 0.2, 10))

    return model_core, model_wing

def chi2_estimate(model_c, model_w):
    chi2 = 0.0

    # GCN circular optical light curves
    for _, nu, nu_bw, t, f, e in circular_processed:
        mc = model_c.flux_density(t, nu * np.ones_like(t))
        mw = model_w.flux_density(t, nu * np.ones_like(t))
        m  = mc.total + mw.total
        chi2 += np.sum((f - m) ** 2 / e ** 2)

    # 7DT SED (single epoch, medium bands)
    for nu, t, f, e in sdt_processed:
        mc = model_c.flux_density(t, nu * np.ones_like(t))
        mw = model_w.flux_density(t, nu * np.ones_like(t))
        m  = mc.total + mw.total
        chi2 += np.sum((f - m) ** 2 / e ** 2)

    # XRT light curves
    for df in xray_data:
        t = df[:, 0]
        f = df[:, 1]   # erg/s/cm², no conversion needed
        e = df[:, 2]   # same unit

        mc = model_c.flux(t, XRT_NU_MIN, XRT_NU_MAX, 10)
        mw = model_w.flux(t, XRT_NU_MIN, XRT_NU_MAX, 10)
        m = mc.total + mw.total
        chi2 += np.sum((f - m) ** 2 / e ** 2)

    return chi2

def log_likelihood(par):
    try:
        phys = to_physical(par)
        model_c, model_w = get_model(**phys)
        chi2 = chi2_estimate(model_c, model_w)
        return -0.5 * chi2
    except Exception:
        return -np.inf

def log_prior(par):
    if np.all(par > pl) and np.all(par < pu):
        phys = to_physical(par)
        if phys["theta_w"] <= phys["theta_c"]:
            return -np.inf
        if phys["E_iso"] <= phys["E_iso_w"]:
            return -np.inf
        if phys["Gamma0"] <= phys["Gamma0_w"]:
            return -np.inf
        return 0.0
    return -np.inf

def log_probability(par):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(par)

ndim     = len(PARAMS)
nwalkers = 10 * ndim 
nsteps   = 30000
nburn    = 5000
npool    = 20

# p0_center = 0.5 * (pl + pu)
# p0 = p0_center + 1e-2 * (pu - pl) * np.random.randn(nwalkers, ndim)
# p0 = np.clip(p0, pl + 1e-6 * (pu - pl), pu - 1e-6 * (pu - pl))

print(f"ndim={ndim}, nwalkers={nwalkers}, npool={npool}, nsteps={nsteps}")

#%%
# if RE_DIR is None:
#     backend = emcee.backends.HDFBackend(f"{RUN_DIR}/mcmc_chain.h5")
#     backend.reset(nwalkers, ndim)  # omit this line to resume

#     with ThreadPoolExecutor(max_workers=npool) as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
#         sampler.run_mcmc(p0, nsteps, progress=True)

#     flat_samples = sampler.get_chain(discard=nburn, thin=1, flat=True)
#     log_probs    = sampler.get_log_prob(discard=nburn, thin=1, flat=True)

#     np.save(f"{RUN_DIR}/mcmc_chain.npy", flat_samples)
#     np.save(f"{RUN_DIR}/mcmc_logprobs.npy", log_probs)

#%%
#################### emcee ####################
# if RE_DIR is None:
#     # Global pre-optimization to seed walkers near the best-fit point.
#     # Invalid regions (prior bounds or ordering constraints) return a large
#     # finite cost so the optimizer can navigate around them.
#     def _neg_log_prob(par):
#         lp = log_probability(par)
#         return -lp if np.isfinite(lp) else 1e10

#     bounds = list(zip(pl, pu))
#     with ThreadPoolExecutor(max_workers=npool) as de_pool:
#         de_result = differential_evolution(
#             _neg_log_prob, bounds,
#             popsize=12, maxiter=40, tol=1e-2,   # these control the optimizer budget
#             polish=True, workers=de_pool.map, updating="deferred",
#             seed=42,
#         )
#     p0_center = de_result.x
#     print(f"Pre-optimization best -log_prob: {de_result.fun:.2f}")

#     # Seed walkers in a tight ball around the optimum
#     p0 = p0_center + 1e-3 * (pu - pl) * np.random.randn(nwalkers, ndim)
#     p0 = np.clip(p0, pl + 1e-6 * (pu - pl), pu - 1e-6 * (pu - pl))

#     backend = emcee.backends.HDFBackend(f"{RUN_DIR}/mcmc_chain.h5")
#     backend.reset(nwalkers, ndim)  # omit this line to resume

#     with ThreadPoolExecutor(max_workers=npool) as pool:
#         moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
#                                         pool=pool, backend=backend, moves=moves)
#         sampler.run_mcmc(p0, nsteps, progress=True)

#     flat_samples = sampler.get_chain(discard=nburn, thin=1, flat=True)
#     log_probs    = sampler.get_log_prob(discard=nburn, thin=1, flat=True)

#     np.save(f"{RUN_DIR}/mcmc_chain.npy", flat_samples)
#     np.save(f"{RUN_DIR}/mcmc_logprobs.npy", log_probs)

#     # ---- Convergence diagnostics ----
#     print(f"\nMean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
#     try:
#         tau = sampler.get_autocorr_time(tol=0)
#         tau_max = np.max(tau)
#         print("Autocorrelation time per parameter:")
#         for name, t in zip(labels, tau):
#             print(f"  {name:10s} {t:8.1f}")
#         print(f"Max tau: {tau_max:.1f}, post-burn-in length: {nsteps - nburn}")
#         print(f"Independent samples per walker ~ {(nsteps - nburn) / tau_max:.1f} (target >= 50)")
#     except Exception as e:
#         print(f"Autocorrelation time estimate failed: {e}")

#     # Trace plot (all walkers overlaid, red line marks the burn-in cut)
#     full_chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
#     _thin_tr = max(1, full_chain.shape[0] // 2000)
#     _steps_tr = np.arange(0, full_chain.shape[0], _thin_tr)
#     fig_tr, axes_tr = plt.subplots(ndim, 1, figsize=(8, 1.8 * ndim), sharex=True)
#     for i in range(ndim):
#         axes_tr[i].plot(_steps_tr, full_chain[::_thin_tr, :, i], color="k", alpha=0.2, lw=0.4)
#         axes_tr[i].axvline(nburn, color="red", ls="--", lw=1.0)
#         axes_tr[i].set_ylabel(labels[i], fontsize=9)
#     axes_tr[-1].set_xlabel("step")
#     fig_tr.tight_layout()
#     fig_tr.savefig(f"{RUN_DIR}/trace_plot.png", dpi=150, bbox_inches="tight")
#     plt.close(fig_tr)
#     print(f"Trace plot saved to {RUN_DIR}/trace_plot.png")

#%%
#################### Dynesty ####################
if RE_DIR is None:
    # ---- Prior transform: unit cube -> parameter space ----
    # Uniform priors via pl + u*(pu-pl). Ordering constraints are enforced by
    # folding each constrained pair (assign max/min), so every proposed point is
    # feasible and the prior matches the uniform-box-with-rejection convention.
    _idx = {name: k for k, name in enumerate(labels)}

    def prior_transform(u):
        x = pl + u * (pu - pl)
        for hi_name, lo_name in (("E_iso", "E_iso_w"),
                                 ("Gamma0", "Gamma0_w"),
                                 ("theta_w", "theta_c")):
            i, j = _idx[hi_name], _idx[lo_name]
            x[i], x[j] = max(x[i], x[j]), min(x[i], x[j])
        return x

    # ---- Likelihood wrapper: finite floor for rare numerical model failures ----
    def loglike(par):
        ll = log_likelihood(par)
        return ll if np.isfinite(ll) else -1e300

    nlive = 1000
    seed  = 1   # change between runs to check evidence reproducibility
    with ThreadPoolExecutor(max_workers=npool) as pool:
        sampler = dynesty.NestedSampler(
            loglike, prior_transform, ndim,
            nlive=nlive, bound="multi", sample="rwalk",
            pool=pool, queue_size=npool,
            rstate=np.random.default_rng(seed),
        )
        sampler.run_nested(checkpoint_file=f"{RUN_DIR}/dynesty.save")
        results = sampler.results

    # ---- Evidence from the per-sample log-weights (independent of results.logz) ----
    logz    = logsumexp(results.logwt)
    weights = np.exp(results.logwt - logz)            # normalized, sums to 1
    H       = np.sum(weights * results.logl) - logz   # information (nats)
    logzerr = np.sqrt(max(H, 0.0) / nlive)

    # ---- Equal-weighted posterior, with aligned log-prob ----
    eq_idx = dyfunc.resample_equal(np.arange(len(results.samples)), weights)
    flat_samples = results.samples[eq_idx]
    log_probs    = results.logl[eq_idx]

    np.save(f"{RUN_DIR}/mcmc_chain.npy", flat_samples)
    np.save(f"{RUN_DIR}/mcmc_logprobs.npy", log_probs)

    print(f"\nln Z = {logz:.3f} +/- {logzerr:.3f}")
    print(f"Equal-weighted samples: {flat_samples.shape[0]}")
    print(f"Effective sample size: {int(results.eff * len(results.samples) / 100)}")
    
#%%
# ===========================================================================
## Best-fit parameters ##
# ===========================================================================
# Load saved chain — allows regenerating plots without re-running MCMC
if RE_DIR is not None:
    flat_samples = np.load(f"{RE_DIR}/mcmc_chain.npy")
    log_probs    = np.load(f"{RE_DIR}/mcmc_logprobs.npy")
    # Recover evidence: from the dynesty checkpoint if present, else set manually
    if not 'logz' in locals() and 'logzerr' in locals():
        try:
            _s = dynesty.NestedSampler.restore(f"{RE_DIR}/dynesty.save")
            logz    = logsumexp(_s.results.logwt)
            logzerr = _s.results.logzerr[-1]
        except Exception:
            logz, logzerr = None, None

    _loaded = _load_params_from_txt(RE_DIR)
    if _loaded is not None:
        PARAMS = _loaded
        labels = [p.name for p in PARAMS]
        print(f"Loaded {len(PARAMS)} params from {RE_DIR}/bestfit_params.txt")
else:
    flat_samples = np.load(f"{RUN_DIR}/mcmc_chain.npy")
    log_probs    = np.load(f"{RUN_DIR}/mcmc_logprobs.npy")
best_par = flat_samples[np.argmax(log_probs)]

# Print chain shape and best log-prob
print(f"\nChain shape: {flat_samples.shape}")
print(f"Best log-prob: {log_probs.max():.2f}")
if 'logz' in locals() and 'logzerr' in locals():
    print(f"ln Z = {logz:.3f} +/- {logzerr:.3f}")

# Print best-fit parameters
best_phys = to_physical(best_par)
model_c_best, model_w_best = get_model(**best_phys)

print("\nBest-fit parameters (physical):")
_THETA_PARAMS = {"theta_c", "theta_w"}
_LOG_PARAMS   = {"E_iso", "E_iso_w"}
for name, val in best_phys.items():
    if name in _LOG_PARAMS:
        print(f"  {name:12s} = {np.log10(val):.4f}  [log10(erg)]")
    elif name in _THETA_PARAMS:
        print(f"  {name:12s} = {np.degrees(val):.4f}  [deg]")
    else:
        print(f"  {name:12s} = {val:.4f}")
        
# After building the best-fit model, print the largest per-point chi2 contributions
def chi2_breakdown(model_c, model_w):
    rows = []
    for name, nu, nu_bw, t, f, e in circular_processed:
        mc = model_c.flux_density(t, nu*np.ones_like(t)).total
        mw = model_w.flux_density(t, nu*np.ones_like(t)).total
        rows += [(name, ti, (fi-mi)**2/ei**2) for ti, fi, ei, mi in zip(t, f, e, mc+mw)]
    for nu, t, f, e in sdt_processed:
        mc = model_c.flux_density(t, nu*np.ones_like(t)).total
        mw = model_w.flux_density(t, nu*np.ones_like(t)).total
        rows += [("SED", t[0], (f[0]-(mc+mw)[0])**2/e[0]**2)]
    df = xray_data[0]
    mc = model_c.flux(df[:,0], XRT_NU_MIN, XRT_NU_MAX, 10).total
    mw = model_w.flux(df[:,0], XRT_NU_MIN, XRT_NU_MAX, 10).total
    rows += [("XRT", ti, (fi-mi)**2/ei**2) for ti, fi, ei, mi in zip(df[:,0], df[:,1], df[:,2], mc+mw)]
    rows.sort(key=lambda r: -r[2])
    for name, t, c in rows[:10]:
        print(f"  {name:5s}  t={t/3600:8.2f} h   chi2={c:7.1f}")
    return rows

chi2_rows = chi2_breakdown(model_c_best, model_w_best)

# %%
# ===========================================================================
## Corner plot ##
# ===========================================================================
if not SKIP_CORNER:
    # Thin to at most 20,000 samples — chain is heavily autocorrelated so
    # subsampling loses almost no information but cuts render time drastically.
    _MAX_CORNER_SAMPLES = 20_000
    if len(flat_samples) > _MAX_CORNER_SAMPLES:
        _idx = np.random.choice(len(flat_samples), _MAX_CORNER_SAMPLES, replace=False)
        _corner_samples = flat_samples[_idx]
    else:
        _corner_samples = flat_samples

    fig = corner.corner(
        _corner_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        truths=np.median(_corner_samples, axis=0),
        truth_color="red",
        bins=20,
        smooth=True,
        fill_contours=True,
        plot_contours=True,
        levels=[0.68, 0.95],
        color="k"
    )
    if RE_DIR is None:
        plt.savefig(f"{RUN_DIR}/corner_plot.png", dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {RUN_DIR}/corner_plot.png")
    else:
        plt.savefig(f"{RE_DIR}/corner_plot.png", dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {RE_DIR}/corner_plot.png")

# %%
# ===========================================================================
## Light curve + SED plot ##
# ===========================================================================
# Chi-squared with best-fit model
best_chi2 = chi2_estimate(model_c_best, model_w_best)
n_data = (sum(len(entry[4]) for entry in circular_processed) +
          len(sdt_processed) +
          xray_data[0].shape[0])
dof = n_data - len(PARAMS)

if RE_DIR is not None:
    param_txt = f"{RE_DIR}/bestfit_params.txt"
else:
    param_txt = f"{RUN_DIR}/bestfit_params.txt"

with open(param_txt, "w") as f:
    # ── Fit Configuration
    f.write("=== Fit Configuration ===\n")
    f.write("Jet Model: tophat (core + ssc) + powerlaw_wing\n")
    if "A_star" in best_phys and ("n0" in best_phys or "n_ism" in best_phys):
        f.write("Medium: stratified medium\n")
    elif "A_star" in best_phys:
        f.write("Medium: Wind\n")
    else:
        f.write("Medium: ISM\n")
    f.write(f"Sampler: emcee (nsteps={nsteps}, nburn={nburn}, npool={npool})\n")
    f.write(f"Chain shape: {flat_samples.shape}\n")
    f.write(f"Best log-prob: {log_probs.max():.4f}\n")
    f.write("\n")

    # ── Data Used
    n_circular = sum(len(entry[3]) for entry in circular_processed)
    f.write("=== Data Used ===\n")
    f.write(f"XRT Data: {xray_data[0].shape[0]} points\n")
    f.write(f"SDT Data: {len(sdt_processed)} points\n")
    f.write(f"Circular Data: {n_circular} points\n")
    if use_i_band:
        f.write(f"i-band Data: {len(iband_processed)} points\n")
    f.write("\n")

    # ── Parameter Setup
    f.write("=== Parameter Setup ===\n")
    for p in PARAMS:
        scale = "log10" if p.log10 else "linear"
        f.write(f'Param("{p.name}", {p.lower:>8g}, {p.upper:>8g}, {scale}),\n')
    f.write("\n")

    # ── Best Fit Statistics
    f.write("=== Best Fit Statistics ===\n")
    f.write(f"Total Data Points (n): {n_data}\n")
    f.write(f"Free Parameters (k): {len(PARAMS)}\n")
    bic = len(PARAMS) * np.log(n_data) + best_chi2
    f.write(f"Best chi^2: {best_chi2:.2f}\n")
    f.write(f"Reduced chi^2/dof: {best_chi2/dof:.2f}\n")
    f.write(f"BIC: {bic:.2f}\n")
    f.write(f"Chain file (npy): {RUN_DIR}/mcmc_chain.npy\n")
    if 'logz' in locals() and 'logzerr' in locals():
        f.write(f"LogZ: {logz:.3f} +/- {logzerr:.3f}\n")
    f.write("\n")

    # ── Best-fit parameters (sampled space, with log10 prefix for log params)
    col_names = [f"log10_{p.name}" if p.log10 else p.name for p in PARAMS]
    header_vals = ["chi^2"] + col_names
    col_w = max(len(h) for h in header_vals) + 2

    f.write("=== Best-fit Parameters ===\n")
    header = f"{'chi^2':>{col_w}}" + "".join(f"{h:>{col_w}}" for h in col_names)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    row = f"{best_chi2:>{col_w}.2f}" + "".join(f"{v:>{col_w}.4f}" for v in best_par)
    f.write(row + "\n")
    f.write("\n")

    # ── Physical values (to_physical), with E_iso in log10 and theta in degrees
    _THETA_PARAMS = {"theta_c", "theta_w"}
    _LOG_PARAMS   = {"E_iso", "E_iso_w"}
    f.write("=== Physical Values (to_physical) ===\n")
    f.write(f"{'Parameter':<16} {'Value':>20}  {'Unit'}\n")
    f.write("-" * 46 + "\n")
    for name, val in best_phys.items():
        if name in _LOG_PARAMS:
            f.write(f"  {name:<14} {np.log10(val):>20.4f}  log10(erg)\n")
        elif name in _THETA_PARAMS:
            f.write(f"  {name:<14} {np.degrees(val):>20.4f}  deg\n")
        else:
            f.write(f"  {name:<14} {val:>20.4f}  \n")
    f.write("\n")

    # ── Largest per-point chi^2 contributions
    f.write("=== Chi^2 Breakdown (Top 10 Contributors) ===\n")
    f.write(f"{'Source':<6} {'t [h]':>10} {'chi^2':>10}\n")
    f.write("-" * 28 + "\n")
    for name, t, c in chi2_rows[:10]:
        f.write(f"{name:<6} {t/3600:>10.2f} {c:>10.1f}\n")

print(f"Best-fit parameters saved to {param_txt}")

#%%
##################### Plotting #####################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
plotted_labels = set()

# ── XRT model
color, _ = BAND_STYLE["XRT"]
if not ONLY_PLOT:
    xrt_mc = model_c_best.flux(LC_T_RANGE, XRT_NU_MIN, XRT_NU_MAX, 10).total
    xrt_mw = model_w_best.flux(LC_T_RANGE, XRT_NU_MIN, XRT_NU_MAX, 10).total
    ax1.plot(LC_T_RANGE / 3600, xrt_mc + xrt_mw, "-",  color=color, lw=1.5)
    ax1.plot(LC_T_RANGE / 3600, xrt_mc,           ":",  color=color, lw=1.2, alpha=0.5)
    ax1.plot(LC_T_RANGE / 3600, xrt_mw,           "-.", color=color, lw=1.2, alpha=0.5)

# ── XRT data (full, unfiltered — the fit itself uses filtered_xrt_data)
ax1.errorbar(
    xrt_data["time"].to_numpy() / 3600,
    xrt_data["flux"].to_numpy(),
    yerr=xrt_err_all,
    fmt=".", color=color, markersize=7,
    markeredgecolor="k", markeredgewidth=0.4, label="X-ray"
)
plotted_labels.add("X-ray")

# ── Optical model (erg/cm²/s, bandwidth-integrated) + data
for (filter_name, nu, nu_bw, *_), (_, t_hrs, flux_erg, err_erg) in zip(circular_processed, circular_plot_data):
    if filter_name not in BAND_STYLE:
        continue
    color, band_label = BAND_STYLE[filter_name]
    
    if not ONLY_PLOT:
        opt_mc = model_c_best.flux(LC_T_RANGE, nu - nu_bw, nu + nu_bw, 10).total
        opt_mw = model_w_best.flux(LC_T_RANGE, nu - nu_bw, nu + nu_bw, 10).total
        ax1.plot(LC_T_RANGE / 3600, opt_mc + opt_mw, "-",  color=color, lw=1.5, alpha=0.8)
        ax1.plot(LC_T_RANGE / 3600, opt_mc,           ":",  color=color, lw=1.2, alpha=0.5)
        ax1.plot(LC_T_RANGE / 3600, opt_mw,           "-.", color=color, lw=1.2, alpha=0.5)

    ax1.errorbar(
        t_hrs, flux_erg, yerr=err_erg,
        fmt=".", color=color, markersize=7,
        markeredgecolor="k", markeredgewidth=0.4,
        label=band_label if band_label not in plotted_labels else "_nolegend_"
    )
    plotted_labels.add(band_label)

if use_i_band:
    for (filter_name, nu, nu_bw, *_), (_, t_hrs, flux_erg, err_erg) in zip(iband_processed, iband_plot_data):
        if filter_name not in BAND_STYLE:
            continue
        color, band_label = BAND_STYLE[filter_name]
        if not ONLY_PLOT:
            i_mc = model_c_best.flux(LC_T_RANGE, nu - nu_bw, nu + nu_bw, 10).total
            i_mw = model_w_best.flux(LC_T_RANGE, nu - nu_bw, nu + nu_bw, 10).total
            ax1.plot(LC_T_RANGE / 3600, i_mc + i_mw, "-",  color=color, lw=1.5, alpha=0.8)
            ax1.plot(LC_T_RANGE / 3600, i_mc,           ":",  color=color, lw=1.2, alpha=0.5)
            ax1.plot(LC_T_RANGE / 3600, i_mw,           "-.", color=color, lw=1.2, alpha=0.5)

        ax1.errorbar(
            t_hrs, flux_erg, yerr=err_erg,
            fmt=".", color=color, markersize=7,
            markeredgecolor="k", markeredgewidth=0.4,
            label=band_label if band_label not in plotted_labels else "_nolegend_"
        )
        plotted_labels.add(band_label)

ax1.plot([], [], "-",  color="k", lw=1.5, label="Total")
ax1.plot([], [], ":",  color="k", lw=1.2, alpha=0.5, label="Core")
ax1.plot([], [], "-.", color="k", lw=1.2, alpha=0.5, label="Wing")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("t [h]")
ax1.set_ylabel(r"$F$ [erg/cm$^2$/s]")
ax1.set_title("Light Curve")
ax1.legend(fontsize=7, loc="upper left")

# Chi-squared text box
if not ONLY_PLOT:
    bic = len(PARAMS) * np.log(n_data) + best_chi2
    stats_text = f"$\\chi^2$: {best_chi2:.1f}\n$\\chi^2$/dof: {best_chi2/dof:.2f}\nBIC: {bic:.1f}"
    if 'logz' in locals() and 'logzerr' in locals():
        stats_text += f"\nLogZ: {logz:.3f} +/- {logzerr:.3f}"
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── SED model at SDT epoch
t_spec   = np.full_like(SPEC_F_RANGE, SDT_SECONDS)
if not ONLY_PLOT:
    spec_mc  = model_c_best.flux_density(t_spec, SPEC_F_RANGE).total.flatten()
    spec_mw  = model_w_best.flux_density(t_spec, SPEC_F_RANGE).total.flatten()
    spec_tot = spec_mc + spec_mw

    ax2.plot(SPEC_F_RANGE, spec_tot, "-",  color="k", lw=1.5,
            label=f"Total at t={SDT_SECONDS/3600:.2f} h")
    ax2.plot(SPEC_F_RANGE, spec_mc,  ":",  color="k", lw=1.2, alpha=0.5, label="Core")
    ax2.plot(SPEC_F_RANGE, spec_mw,  "-.", color="k", lw=1.2, alpha=0.5, label="Wing")

    # Spectral index annotations
    log_nu = np.log10(SPEC_F_RANGE)
    beta_tot  = np.polyfit(log_nu, np.log10(spec_tot), 1)[0]
    idx_t = int(len(SPEC_F_RANGE) * 0.3)
    ax2.text(SPEC_F_RANGE[idx_t], spec_tot[idx_t] * 0.6,
            f"$\\beta \\approx {beta_tot:.2f}$", color="k", fontsize=9, ha="center", va="top")

    beta_core = np.polyfit(log_nu, np.log10(spec_mc), 1)[0]
    idx_c = int(len(SPEC_F_RANGE) * 0.5)
    ax2.text(SPEC_F_RANGE[idx_c], spec_mc[idx_c] * 0.6,
            f"$\\beta_c \\approx {beta_core:.2f}$", color="k", fontsize=9, ha="center", va="top")

    beta_wing = np.polyfit(log_nu, np.log10(spec_mw), 1)[0]
    idx_w = int(len(SPEC_F_RANGE) * 0.7)
    ax2.text(SPEC_F_RANGE[idx_w], spec_mw[idx_w] * 0.6,
            f"$\\beta_w \\approx {beta_wing:.2f}$", color="k", fontsize=9, ha="center", va="top")

# ── 7DT SED data
ax2.errorbar(
    sdt_data["frequency_Hz"],
    sdt_data["flux_mJy"] * mJy,
    yerr=sdt_data["flux_error_mJy"] * mJy,
    xerr=sdt_data["frequency_error_Hz"],
    fmt="o", color="purple", markersize=7,
    markeredgecolor="k", markeredgewidth=0.4, label="7DT SED"
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\nu$ [Hz]")
ax2.set_ylabel(r"$F_\nu$ [erg/cm$^2$/s/Hz]")
ax2.set_title(f"7DT SED at t={SDT_SECONDS/3600:.2f} h")
ax2.legend(fontsize=7)

if ONLY_PLOT:
    lc_dir = f"{PLOT_DIR}/dataonly_lc.png"
elif RE_DIR is not None:
    lc_dir = f"{RE_DIR}/bestfit_lc.png"
else:
    lc_dir = f"{RUN_DIR}/bestfit_lc.png"
plt.tight_layout()
plt.savefig(lc_dir, dpi=300)
plt.close()
print(f"Best-fit plot saved to {lc_dir}")

# %%
