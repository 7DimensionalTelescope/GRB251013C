import numpy as np
import pandas as pd
from .const import FILTER_INFO, MAG_ERROR_FLOOR
from astropy import constants as const
from astropy import units as u

def mag_to_flux_mJy(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("DataFrame expected")

    mag = df["magnitude"].to_numpy(float) if "magnitude" in df.columns else df["Mag"].to_numpy(float)
    mag_err = df["mag_error"].to_numpy(float) if "mag_error" in df.columns else df["Error"].to_numpy(float)
    # Calibration floor; a missing error becomes the floor.
    mag_err = np.where(np.isfinite(mag_err), np.maximum(mag_err, MAG_ERROR_FLOOR), MAG_ERROR_FLOOR)

    # Per-band zero point: FILTER_INFO declares Rc/Ic/V/R/J as Vega-system
    # photometry (e.g. GCN 42333 Leavitt: Lupton-2005 conversion => Vega).
    # Using the AB ZP (3631 Jy) for them over-brightens Rc by 0.18 mag and
    # Ic by 0.38 mag.
    filters = df["filter"].tolist() if "filter" in df.columns else [None] * len(mag)

    if "wavelength" in df.columns:
        wavelengths = df["wavelength"].to_numpy(float)
        bandwidths = df["filter_width"].to_numpy(float)
        unit = "AA"
        flx_list, flx_err_list = [], []
        for m, e, filt in zip(mag, mag_err, filters):
            info = FILTER_INFO.get(filt) if filt is not None else None
            zp = info["vega_zero_point_jy"] if info is not None and info.get("system") == "Vega" else 3631
            f, fe = _mag_to_flux_mJy(m, e, zero_point=zp)
            flx_list.append(f)
            flx_err_list.append(fe)
    else:
        raise ValueError("wavelength column not found")

    wl_arr = np.asarray(wavelengths)
    df["frequency_Hz"] = unit_conversion(wl_arr, unit, "Hz")
    df["frequency_Hz_error"] = (
        unit_conversion(wl_arr - np.asarray(bandwidths) / 2.0, unit, "Hz")
        - df["frequency_Hz"]
    )
    df["flux_mJy"] = flx_list
    df["flux_mJy_error"] = flx_err_list 
    return df

def _mag_to_flux_mJy(mag, mag_err=None, zero_point=3631):
    flux_jy = zero_point*10**(-0.4 * mag) * 1e3
    if mag_err is not None:
        flux_jy_err = flux_jy * (np.log(10) * 0.4) * mag_err
    else:
        flux_jy_err = None
    return flux_jy, flux_jy_err

def heliocentric_correction_seconds(date_obs, ra=None, dec=None):
    """HJD - JD (seconds) for the burst position at the given UTC time(s).

    Positive means the heliocentric time stamp is later than the geocentric
    one, i.e. an HJD mistakenly read as JD/UTC is late by this amount."""
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    if ra is None or dec is None:
        from .const import RA, DEC
        ra, dec = RA, DEC
    t = Time(pd.to_datetime(date_obs).to_numpy(), scale="utc", location=EarthLocation.from_geodetic(0, 0, 0))
    return t.light_travel_time(SkyCoord(ra, dec, unit="deg"), kind="heliocentric").to_value("s")


def unit_conversion(value, input_unit, output_unit):
    return u.Quantity(value, input_unit).to(output_unit, equivalencies=u.spectral()).value

def mJy_to_erg_cm2_s(flux_mJy, nu_Hz):
    return flux_mJy * nu_Hz * 1e-26

def mJy_to_erg_cm2_s_Hz(flux_mJy):
    return flux_mJy * 1e-26

def mask_data(x_data, y_data, x_data_error=None, y_data_error=None):
    mask_y = np.isfinite(y_data) & (y_data > 0)
    mask_x = np.isfinite(x_data) & (x_data > 0)
    if x_data_error is not None:
        mask_x_error = np.isfinite(x_data_error) & (x_data_error > 0)
    else:
        mask_x_error = True
    if y_data_error is not None:
        mask_y_error = np.isfinite(y_data_error) & (y_data_error > 0)
    else:
        mask_y_error = True
    mask = mask_y & mask_x & mask_x_error & mask_y_error
    return mask

def filter_to_wavelength(filter_name):
    if isinstance(filter_name, str):
        if filter_name in FILTER_INFO:
            return FILTER_INFO[filter_name]["central_wavelength_nm"] * 10 # in AA
        elif isinstance(filter_name, str) and filter_name.startswith("m"):
            return float(filter_name.replace("m", ""))
        else:
            print(f"Invalid filter name: {filter_name}")
            return 0
            
    elif isinstance(filter_name, list):
        return [filter_to_wavelength(filt) for filt in filter_name]
    elif isinstance(filter_name, pd.Series):
        return [filter_to_wavelength(filt) for filt in filter_name.to_list()]
    else:
        print(f"Invalid input type: {type(filter_name)}")
        return 0

def filter_width(filter_name):
    if isinstance(filter_name, str):
        if filter_name in FILTER_INFO:
            return FILTER_INFO[filter_name]["bandwidth_nm"] * 10 # in AA
        elif isinstance(filter_name, str) and filter_name.startswith("m"):
            return 250
        else:
            print(f"Invalid filter name: {filter_name}")
            return 0
    elif isinstance(filter_name, list):
        return [filter_width(filt) for filt in filter_name]
    elif isinstance(filter_name, pd.Series):
        return [filter_width(filt) for filt in filter_name.to_list()]
    else:
        print(f"Invalid input type: {type(filter_name)}")
        return 0


def flux_error(df):
    return np.maximum(
        np.abs(df["flux_high"].to_numpy(float)),
        np.abs(df["flux_low"].to_numpy(float)),
    )


def seconds_from_trigger(date_obs):
    """Convert date_obs string to seconds from trigger"""
    from datetime import datetime
    from .const import TRIGGER_TIME
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.strptime(str(date_obs), fmt) - TRIGGER_TIME).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date_obs format: {date_obs}")


def mjd_to_seconds_from_trigger(mjd):
    """Convert MJD (scalar or array) to seconds from trigger."""
    from astropy.time import Time
    from .const import TRIGGER_TIME
    trigger_mjd = Time(TRIGGER_TIME.strftime("%Y-%m-%dT%H:%M:%S")).mjd
    return (np.asarray(mjd, dtype=float) - trigger_mjd) * 86400.0


def normalize_filter_name(filter_name):
    """Map catalog filter labels (e.g. sdssi) onto FILTER_INFO keys (e.g. i)."""
    name = str(filter_name).strip()
    if name.lower().startswith("sdss") and len(name) > 4:
        return name[4:]
    return name


def model_array(model_output):
    if hasattr(model_output, "total"):
        model_output = model_output.total
    if hasattr(model_output, "sync"):
        model_output = np.asarray(model_output.sync) + np.asarray(model_output.ssc)
    return np.asarray(model_output).squeeze()
