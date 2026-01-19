import numpy as np
from astropy import units as u
from astropy import constants as const

import astropy.units as u
from astropy.coordinates import SpectralCoord
import pandas as pd
from .const import FILTER_INFO

def mag_to_flux_jy(mag, filt):
    if isinstance(mag, pd.Series):
        mag = mag.to_numpy("float")
        
    info = FILTER_INFO[filt]
    if info["system"] == "AB":
        return 3631 * 10**(-0.4 * mag)
    elif info["system"] == "Vega":
        return info["vega_zero_point_jy"] * 10**(-0.4 * mag)
    else:
        raise ValueError("Instrumental filter: zeropoint undefined")
