import numpy as np
import pandas as pd
from .const import FILTER_INFO
from astropy import constants as const
from astropy import units as u

def mag_to_flux_mJy(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("DataFrame expected")

    mag = df["magnitude"].to_numpy(float) if "magnitude" in df.columns else df["Mag"].to_numpy(float)
    mag_err = df["mag_error"].to_numpy(float) if "mag_error" in df.columns else df["Error"].to_numpy(float)

    if "wavelength" in df.columns:
        wavelengths = df["wavelength"].to_numpy(float)
        bandwidths = df["filter_width"].to_numpy(float)
        unit = "AA"
        flx_list, flx_err_list = [], []
        for m, e in zip(mag, mag_err):
            f, fe = _mag_to_flux_mJy(m, e)
            flx_list.append(f)
            flx_err_list.append(fe)
    elif "Filter" in df.columns:
        wavelengths, bandwidths, flx_list, flx_err_list = [], [], [], []
        unit = "nm"
        for m, e, filt in zip(mag, mag_err, df["Filter"]):
            f, fe, wl, bw = _flux_from_filter(m, e, filt)
            flx_list.append(f)
            flx_err_list.append(fe)
            wavelengths.append(wl)
            bandwidths.append(bw)
    else:
        raise ValueError("Wavelength or Filter column not found")

    wl_arr = np.asarray(wavelengths)
    df["frequency_Hz"] = _wavelength_to_frequency(wl_arr, wavelength_unit=unit).value
    df["frequency_error_Hz"] = (
        _wavelength_to_frequency(wl_arr - np.asarray(bandwidths) / 2.0, wavelength_unit=unit).value
        - df["frequency_Hz"]
    )
    df["flux_mJy"] = flx_list
    df["flux_error_mJy"] = flx_err_list 
    return df

def _flux_from_filter(mag, mag_err, filter_name):
    """Return (flux_jy, flux_err_jy, wavelength_nm, bandwidth_nm) from filter name."""
    if filter_name in FILTER_INFO:
        info = FILTER_INFO[filter_name]
        wl, bw = info["central_wavelength_nm"], info["bandwidth_nm"]
        if info["system"] == "AB":
            zero_point = 3631
        elif info["system"] == "Vega":
            zero_point = info["vega_zero_point_jy"]
        elif info["system"] == "instrumental":
            return None, None, wl, bw
        else:
            return None, None, wl, bw
            
        flx, flx_err = _mag_to_flux_mJy(mag, mag_err, zero_point=zero_point)
        
        return flx, flx_err, wl, bw
    if isinstance(filter_name, str) and filter_name.startswith("m"):
        wl = float(filter_name.replace("m", ""))
        
        flx, flx_err = _mag_to_flux_mJy(mag, mag_err)
        return flx, flx_err, wl, 250.0
    return None, None, np.nan, np.nan

def _mag_to_flux_mJy(mag, mag_err=None, zero_point=3631):
    flux_jy = zero_point*10**(-0.4 * mag) * 1e3
    if mag_err is not None:
        flux_jy_err = flux_jy * (np.log(10) * 0.4) * mag_err
    else:
        flux_jy_err = None
    return flux_jy, flux_jy_err

def _wavelength_to_frequency(wavelength, wavelength_unit='AA'):
    wavelength = np.asarray(wavelength)
    safe_wl = np.where(wavelength == 0, np.nan, wavelength)
    frequency = (const.c / u.Quantity(safe_wl, wavelength_unit)).to(u.Hz)
    return frequency

