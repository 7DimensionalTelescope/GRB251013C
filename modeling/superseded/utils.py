from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from VegasAfterglow import ParamDef, Scale
from VegasAfterglow.extinction import BUILTIN_LAWS


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


XRT_EXCLUDE_TIME_RANGE = (3e3, 1e4)
XRT_FLARE_START_TIME = 3e3
XRT_FLARE_END_TIME = 1e4
HOST_AV_LOG10_MEAN = -0.82
HOST_AV_LOG10_SIGMA = 0.41
C_CM_PER_S = 2.99792458e10
LN10_OVER_2P5 = 0.4 * np.log(10.0)


def default_nwalkers(ndim):
    return max(4 * ndim, 32)


def top_k_samples(samples, log_probs, top_k):
    order = np.argsort(log_probs)[::-1]
    samples_sorted = samples[order]
    log_probs_sorted = log_probs[order]
    keep = []
    seen = set()
    for idx, sample in enumerate(np.round(samples_sorted, 12)):
        key = tuple(sample)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
        if len(keep) >= top_k:
            break
    return samples_sorted[keep], log_probs_sorted[keep]


def xrt_flux_error(df):
    return np.maximum(
        np.abs(df["flux_high"].to_numpy(float)),
        np.abs(df["flux_low"].to_numpy(float)),
    )


def host_extinction_attenuation(nu_hz, a_v, redshift, profile="smc"):
    if a_v == 0:
        return np.ones_like(np.asarray(nu_hz, dtype=float))

    law = BUILTIN_LAWS[profile]
    lambda_rest_cm = C_CM_PER_S / np.asarray(nu_hz, dtype=float) / (1.0 + redshift)
    k_lambda = law(lambda_rest_cm)
    return np.exp(-a_v * LN10_OVER_2P5 * k_lambda)


def read_labels(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_xrt_spectral_index(data_dir=None):
    """Load XRT spectral index measurements from photon index data.
    
    Returns:
        dict with keys: time, time_low, time_high, beta, beta_err_low, beta_err_high
        
    Note:
        The CSV file contains photon index Γ. 
        Converted to spectral index β (Granot & Sari convention) where F_ν ∝ ν^β.
        For synchrotron: β = 1 - Γ (negative values, flux decreases with frequency).
    """
    import pandas as pd
    if data_dir is None:
        from pathlib import Path
        import os
        data_dir = Path(os.path.dirname(__file__)).parent / "data"
    
    df = pd.read_csv(
        data_dir / "xrt_index.csv",
        names=["time", "time_err_high", "time_err_low", "gamma", "gamma_err_high", "gamma_err_low"]
    )
    
    # Convert photon index (Γ) to spectral index (β = 1 - Γ, Granot & Sari convention)
    beta = 1.0 - df["gamma"].to_numpy(float)
    
    return {
        "time": df["time"].to_numpy(float),
        "time_low": df["time_err_low"].abs().to_numpy(float),
        "time_high": df["time_err_high"].to_numpy(float),
        "beta": beta,
        "beta_err_low": df["gamma_err_low"].abs().to_numpy(float),
        "beta_err_high": df["gamma_err_high"].to_numpy(float),
    }


def compute_break_frequencies(params, z, t_obs):
    """Compute synchrotron break frequencies from Granot & Sari 2001.
    
    Args:
        params: dict with keys E_iso, Gamma0, n_ism, eps_e, eps_B, p
        z: redshift
        t_obs: observer time in seconds
        
    Returns:
        dict with keys: nu_m, nu_c (in Hz)
        
    Reference:
        Granot & Sari 2001, Table 2 (arxiv:astro-ph/0108027v1)
        Equations for ISM (k=0) case
    """
    # Extract parameters
    E_52 = params.get("E_iso", 1e52) / 1e52
    n_0 = params.get("n_ism", 1.0)
    eps_e = params.get("eps_e", 0.1)
    eps_B = params.get("eps_B", 0.01)
    p = params.get("p", 2.2)
    
    # Time in days
    t_days = t_obs / 86400.0
    
    # ISM case (k=0) from Table 2, break 2 (νm)
    # Formula for p=2.5 case, adjusted for general p
    nu_m_prefactor = 3.73 * (p - 0.67) * 1e15  # Hz, approximate p-dependence
    nu_m = nu_m_prefactor * ((1 + z) ** 0.5) * (E_52 ** 0.5) * (eps_e ** 2) * (eps_B ** 0.5) * (t_days ** (-3/2))
    
    # ISM case (k=0) from Table 2, break 3 (νc)
    # Formula for cooling frequency
    nu_c_prefactor = 6.37 * (p - 0.46) * 1e13 * np.exp(-1.16 * p)  # Hz
    nu_c = nu_c_prefactor * ((1 + z) ** (-0.5)) * (eps_B ** (-3/2)) * (n_0 ** (-1)) * (E_52 ** (-0.5)) * (t_days ** (-0.5))
    
    return {
        "nu_m": nu_m,
        "nu_c": nu_c,
    }


def compute_p_prior_from_spectral_index(xrt_index_data, cooling_regime="slow"):
    """Compute gaussian prior on p from XRT spectral index measurements.
    
    Args:
        xrt_index_data: Dict with 'beta' and error arrays from load_xrt_spectral_index()
        cooling_regime: 'slow', 'fast', or 'both'
    
    Returns:
        (mean_p, sigma_p) tuple for Gaussian prior on p
        
    Note:
        Assumes a fixed cooling regime for the XRT band (0.3-10 keV).
        'slow' cooling (β = (p-1)/2) is the default for this GRB.
    """
    beta_values = xrt_index_data["beta"]
    beta_errs = np.maximum(xrt_index_data["beta_err_low"], xrt_index_data["beta_err_high"])
    
    # Weight by inverse variance
    weights = 1.0 / (beta_errs ** 2)
    beta_mean = np.average(beta_values, weights=weights)
    beta_std = np.sqrt(1.0 / np.sum(weights))
    
    # Convert β (G&S convention, negative) to p based on cooling regime
    if cooling_regime == "slow":
        # β = (1-p)/2  =>  p = 1 - 2β (G&S Table 2, slow cooling between breaks)
        p_mean = 1.0 - 2.0 * beta_mean
        p_sigma = 2.0 * beta_std
    elif cooling_regime == "fast":
        # β = -p/2  =>  p = -2β (G&S Table 2, fast cooling above nu_c)
        p_mean = -2.0 * beta_mean
        p_sigma = 2.0 * beta_std
    elif cooling_regime == "both":
        # Average of both relations
        p_slow = 1.0 - 2.0 * beta_mean
        p_fast = -2.0 * beta_mean
        p_mean = 0.5 * (p_slow + p_fast)
        p_sigma = 2.0 * beta_std
    else:
        raise ValueError(f"Unknown cooling regime: {cooling_regime}")
    
    return p_mean, p_sigma


def latest_result_dir(base_dir, prefix):
    candidates = []
    for path in Path(base_dir).glob(f"{prefix}*"):
        if (path / "top_k_params.npy").exists() and (path / "labels.txt").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No usable {prefix}* result directory found in {base_dir}")
    return sorted(candidates)[-1]


def plot_corner(outdir, labels=None, max_samples=20000, seed=42):
    outdir = Path(outdir)
    samples = np.load(outdir / "samples.npy")
    samples = samples.reshape(-1, samples.shape[-1])
    log_probs = np.load(outdir / "log_probs.npy").reshape(-1)
    if labels is None:
        labels = read_labels(outdir / "labels.txt")

    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(log_probs)
    samples = samples[finite]
    log_probs = log_probs[finite]

    # Best-fit (max log-prob) sample, marked with red crosshairs
    best_fit = samples[np.argmax(log_probs)]

    if len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        samples = samples[rng.choice(len(samples), max_samples, replace=False)]

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        bins=30,
        smooth=True,
        truths=best_fit,
        truth_color="red",
    )
    fig.savefig(outdir / "corner_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def set_log_y_limits(ax, *values, lower_factor=0.5, upper_factor=2.0):
    positive = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array) & (array > 0)]
        if len(array):
            positive.append(array)
    if not positive:
        return

    combined = np.concatenate(positive)
    ax.set_ylim(combined.min() * lower_factor, combined.max() * upper_factor)


def model_array(model_output):
    if hasattr(model_output, "total"):
        model_output = model_output.total
    if hasattr(model_output, "sync"):
        model_output = np.asarray(model_output.sync) + np.asarray(model_output.ssc)
    return np.asarray(model_output).squeeze()
