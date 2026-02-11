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
    df["frequency_Hz"] = unit_conversion(wl_arr, unit, "Hz")
    df["frequency_error_Hz"] = (
        unit_conversion(wl_arr - np.asarray(bandwidths) / 2.0, unit, "Hz")
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

def unit_conversion(value, input_unit, output_unit):
    return u.Quantity(value, input_unit).to(output_unit, equivalencies=u.spectral()).value

def mJy_to_erg_cm2_s(flux_mJy, nu_Hz):
    return flux_mJy * nu_Hz * 1e-26

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