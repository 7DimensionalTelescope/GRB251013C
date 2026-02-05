import numpy as np
from .const import FILTER_INFO
from astropy import constants as const
from astropy import units as u

def wavelength_to_frequency(wavelength, wavelength_unit='AA'):
    wavelength = u.Quantity(wavelength, wavelength_unit)
    return (const.c / wavelength).to(u.Hz)

def mag_to_flux_jy(df, add_error=False):

    mag_list = df["Mag"].to_numpy("float")
    filt_list = df["Filter"]
    if add_error:
        mag_err_list = df["Error"].to_numpy("float")
    else:
        mag_err_list = np.zeros_like(mag_list)
    flx_list = []
    flx_err_list = []
    for mag, filt, mag_err in zip(mag_list, filt_list, mag_err_list):        
        info = FILTER_INFO[filt]
        if info["system"] == "AB":
            flx = 3631 * 10**(-0.4 * mag)
        elif info["system"] == "Vega":
            flx = info["vega_zero_point_jy"] * 10**(-0.4 * mag)
        else:
            raise ValueError("Instrumental filter: zeropoint undefined")
        if add_error:
            flx_err = _mag_err_to_flux_err_jy(mag_err, flx)
            flx_list.append(flx)
            flx_err_list.append(flx_err)
        else:
            flx_list.append(flx)

    return np.array(flx_list), np.array(flx_err_list)

def _mag_err_to_flux_err_jy(mag_err, flux_jy):
    return flux_jy * (np.log(10) * 0.4) * mag_err