"""XRT spectral-index tooling (Granot & Sari 2001).

Derives the observed spectral index beta = 1 - Gamma from the XRT photon index,
computes synchrotron break frequencies, and turns the measured index into a
Gaussian prior on the electron index p.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .const import DATA_DIR


def load_xrt_spectral_index(data_dir=None):
    """Load XRT spectral index measurements from photon index data.

    Returns:
        dict with keys: time, time_low, time_high, beta, beta_err_low, beta_err_high

    Note:
        The CSV file contains photon index Γ.
        Converted to spectral index β (Granot & Sari convention) where F_ν ∝ ν^β.
        For synchrotron: β = 1 - Γ (negative values, flux decreases with frequency).
    """
    if data_dir is None:
        data_dir = Path(DATA_DIR)
    else:
        data_dir = Path(data_dir)

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
