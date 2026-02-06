import numpy as np
import pandas as pd
from .const import FILTER_INFO
from astropy import constants as const
from astropy import units as u


def mag_to_flux_jy(df, add_error=True):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("DataFrame expected")

    mag_list = df["magnitude"].to_numpy("float") if "magnitude" in df.columns else df["Mag"].to_numpy("float")
    
    if add_error: 
        mag_err_list = df["mag_error"].to_numpy("float") if "mag_error" in df.columns else df["Error"].to_numpy("float")
    
    if "wavelength" in df.columns:
        wavelengths = df["wavelength"].to_numpy("float")
        bandwidths = df["filter_width"].to_numpy("float")
        exist_wavelength = True
    elif "Filter" in df.columns:
        filter_list = df["Filter"]
        wavelengths = []
        bandwidths = []
        exist_wavelength = False
    else:
        raise ValueError("Wavelength or Filter column not found")

    flx_list = []
    flx_err_list = []
    for i, (mag, mag_err) in enumerate(zip(mag_list, mag_err_list)):        
        if exist_wavelength:
            flx, flx_err = _mag_to_flux_jy(mag, mag_err if add_error else None)
            unit = 'AA'
        else:
            if filter_list[i] in FILTER_INFO.keys():
                info = FILTER_INFO[filter_list[i]]
                if info["system"] == "AB":
                    flx = 3631 * 10**(-0.4 * mag)
                    if add_error: flx_err = _mag_err_to_flux_err_jy(mag_err, flx)
                elif info["system"] == "Vega":
                    flx = info["vega_zero_point_jy"] * 10**(-0.4 * mag)
                    if add_error: flx_err = _mag_err_to_flux_err_jy(mag_err, flx)
                elif info["system"] == "instrumental":
                    flx = None
                    flx_err = None
                
                wavelength = info["central_wavelength_nm"]
                bandwidth = info["bandwidth_nm"]
                unit = 'nm'
                
            elif isinstance(filter_list[i], str) and filter_list[i].startswith("m"):
                wavelength = float(filter_list[i].replace("m", ""))
                bandwidth = 250
                flx, flx_err = _mag_to_flux_jy(mag, mag_err if add_error else None)
                unit = 'nm'
            wavelengths.append(wavelength)
            bandwidths.append(bandwidth)
                        
        flx_list.append(flx)
        flx_err_list.append(flx_err)
            
        
    df["freqeuncy_Hz"] = _wavelength_to_frequency(wavelengths, wavelength_unit=unit).value
    df["frequency_error_Hz"] = _wavelength_to_frequency(np.array(wavelengths)-np.array(bandwidths)/2., wavelength_unit=unit).value - df["freqeuncy_Hz"]
    df["flux_mJy"] = flx_list
    df["flux_error_mJy"] = flx_err_list
    return df

def _mag_err_to_flux_err_jy(mag_err, flux_jy):
    return flux_jy * (np.log(10) * 0.4) * mag_err

def _mag_to_flux_jy(mag, mag_err=None):
    flux_jy = 10**(-0.4 * (mag - 23.9))
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

