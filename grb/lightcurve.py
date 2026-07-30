import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table

from .io import read_data
from .utils import mag_to_flux_mJy, unit_conversion

def prepare_lc_data(df, mag_to_flux=False, filter_name=None):
    tab = Table.from_pandas(df)[["Time", "Mag", "Error"]]
    tab.rename_columns(["Time", "Mag", "Error"], ["t", "y", "y_err"])
    if mag_to_flux and filter_name is not None:
        tab["y"] = mag_to_flux_mJy(tab["y"], filter_name)
        #tab["y_err"] = mag_to_flux_jy(tab["y_err"], filter_name)

    tab.add_column(np.zeros(len(tab)), name="t_err")
    tab.add_column(np.zeros(len(tab)), name="upper_limit")
    tab["upper_limit"] = [">" in str(v) for v in tab["y"]]
    return tab


def plot_lightcurve(data, ax=None, ls="", invert_yaxis = False, upper_limits=None, label=None, **kwargs):
    
    if ax is None:
        ax = plt.gca()

    y_label = kwargs.pop("y_label", "Flux")
    x_label = kwargs.pop("x_label", "Time since trigger (hours)")
    x_scale = kwargs.pop("x_scale", "log")
    y_scale = kwargs.pop("y_scale", "log")

    upper_limits = data["upper_limit"]
    
    ax.errorbar(data["t"][~upper_limits].astype(float), data["y"][~upper_limits].astype(float), yerr=data["y_err"][~upper_limits].astype(float), ls=ls, label=label, **kwargs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if label is not None:
        ax.legend()

    if invert_yaxis:
        ax.invert_yaxis()

    ax.grid(which="major", axis="both", linestyle="-", color="gray", alpha=0.8, lw=0.1)
    ax.grid(which="minor", axis="both", linestyle="--", color="gray", alpha=0.5, lw=0.1)

    return ax


def _smooth_flux_with_power_law_fit(time, flux, window, min_points=3):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    smoothed_flux = flux.copy()
    slopes = np.full(len(flux), np.nan)
    half_window = window // 2

    for idx in range(len(flux)):
        start = max(0, idx - half_window)
        end = min(len(flux), idx + half_window + 1)
        time_window = time[start:end]
        flux_window = flux[start:end]
        valid = np.isfinite(time_window) & np.isfinite(flux_window) & (time_window > 0) & (flux_window > 0)

        if valid.sum() < min_points:
            continue

        slope, intercept = np.polyfit(np.log10(time_window[valid]), np.log10(flux_window[valid]), 1)
        smoothed_flux[idx] = 10 ** (intercept + slope * np.log10(time[idx]))
        slopes[idx] = slope

    return smoothed_flux, slopes


def _fit_power_law_flux(time, flux):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux) & (time > 0) & (flux > 0)

    if valid.sum() < 2:
        raise ValueError("At least two valid points are required to fit a power law")

    slope, intercept = np.polyfit(np.log10(time[valid]), np.log10(flux[valid]), 1)
    fitted_flux = 10 ** (intercept + slope * np.log10(time))
    return fitted_flux, slope, intercept


def extrapolate_xrt_flux_density(
    wavelength,
    wavelength_unit="AA",
    output_unit="mJy",
    xrt_data=None,
    xrt_index_data=None,
    index_type="photon",
    nu_min=None,
    nu_max=None,
    exclude_time_range=(3e3, 1e4),
    smooth_window=None,
    smooth_method="median",
    smooth_columns=("index",),
    flux_smooth_window=None,
    flux_smooth_min_points=3,
    use_mean_index=False,
    fit_flux_power_law=False,
):
    """Extrapolate XRT integrated flux to a flux density at a target wavelength.

    The XRT flux is treated as the energy flux integrated over 0.3-10 keV.
    The spectrum is assumed to be F_nu = K * nu^(-beta). By default the XRT
    index column is interpreted as photon index Gamma, so beta = Gamma - 1.
    """
    if xrt_data is None:
        xrt_data = read_data("xrt")
    if xrt_index_data is None:
        xrt_index_data = read_data("xrt_index")

    if nu_min is None:
        nu_min = unit_conversion(0.3, "keV", "Hz")
    if nu_max is None:
        nu_max = unit_conversion(10, "keV", "Hz")

    target_nu = unit_conversion(wavelength, wavelength_unit, "Hz")
    df = pd.merge(
        xrt_data,
        xrt_index_data[["time", "index", "index_high", "index_low"]],
        on="time",
        how="inner",
    ).sort_values("time").reset_index(drop=True)

    if exclude_time_range is not None:
        df = df[~df["time"].between(exclude_time_range[0], exclude_time_range[1])]
        df = df.reset_index(drop=True)

    if smooth_window is not None and smooth_window > 1:
        if isinstance(smooth_columns, str):
            smooth_columns = [smooth_columns]

        for column in smooth_columns:
            if column not in df.columns:
                raise ValueError(f"Column not found for smoothing: {column}")

            rolling = df[column].rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            )
            if smooth_method == "median":
                df[column] = rolling.median()
            elif smooth_method == "mean":
                df[column] = rolling.mean()
            else:
                raise ValueError("smooth_method must be 'median' or 'mean'")

    if use_mean_index:
        df["index_original"] = df["index"]
        df["index"] = df["index"].mean()

    if fit_flux_power_law:
        df["flux_original"] = df["flux"]
        fitted_flux, slope, intercept = _fit_power_law_flux(df["time"], df["flux"])
        df["flux"] = fitted_flux
        df["flux_fit_slope"] = slope
        df["flux_fit_intercept"] = intercept

    if flux_smooth_window is not None and flux_smooth_window > 1:
        if "flux_original" not in df.columns:
            df["flux_original"] = df["flux"]
        df["flux"], df["flux_smooth_slope"] = _smooth_flux_with_power_law_fit(
            df["time"],
            df["flux"],
            flux_smooth_window,
            min_points=flux_smooth_min_points,
        )

    flux = df["flux"].to_numpy(float)
    index = df["index"].to_numpy(float)
    if index_type == "photon":
        beta = index - 1.0
    elif index_type == "spectral":
        beta = index
    else:
        raise ValueError("index_type must be 'photon' or 'spectral'")

    integral = np.where(
        np.isclose(beta, 1.0),
        np.log(nu_max / nu_min),
        (nu_max ** (1.0 - beta) - nu_min ** (1.0 - beta)) / (1.0 - beta),
    )
    norm = flux / integral
    flux_density = norm * target_nu ** (-beta)

    flux_error = np.maximum(
        np.abs(df["flux_high"].to_numpy(float)),
        np.abs(df["flux_low"].to_numpy(float)),
    )
    flux_for_error = df["flux_original"].to_numpy(float) if "flux_original" in df.columns else flux
    flux_density_error = flux_density * flux_error / flux_for_error

    if output_unit == "mJy":
        flux_density = flux_density / 1e-26
        flux_density_error = flux_density_error / 1e-26
    elif output_unit != "cgs":
        raise ValueError("output_unit must be 'mJy' or 'cgs'")

    result = df[["time", "time_high", "time_low"]].copy()
    result["wavelength"] = wavelength
    result["frequency_Hz"] = target_nu
    result["beta"] = beta
    result["flux_density"] = flux_density
    result["flux_density_error"] = flux_density_error
    result["flux_density_unit"] = output_unit
    if "index_original" in df.columns:
        result["index_original"] = df["index_original"]
    if "flux_smooth_slope" in df.columns:
        result["flux_smooth_slope"] = df["flux_smooth_slope"]
    if "flux_fit_slope" in df.columns:
        result["flux_fit_slope"] = df["flux_fit_slope"]
        result["flux_fit_intercept"] = df["flux_fit_intercept"]
    return result
