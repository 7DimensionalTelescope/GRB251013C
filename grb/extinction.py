import numpy as np

from astropy import units as u
from astropy.io import ascii as asc
from astropy.coordinates import SkyCoord

from scipy.interpolate import interp1d
from .const import DATA_DIR

import extinction
from dustmaps.sfd import SFDQuery
import pandas as pd
import copy

def correct_galactic_extinction(df, magnitude_column="magnitude", wavelength_column="wavelength", **kwargs) -> pd.DataFrame:
    """
    Corrects the magnitude data for galactic extinction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the magnitude data with columns for wavelength and magnitude.
    magnitude_column : str, optional
        The name of the column containing the magnitude data. Default is "magnitude".
    wavelength_column : str, optional
        The name of the column containing the wavelength data. Default is "wavelength".
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the corrected magnitude data.
    """
    if "gal_corrected" in df.columns and df["corrected"].all():
        print("Extinction already corrected")
        return df
    elif "gal_corrected" in df.columns and df["gal_corrected"].any():
        print("Extinction partially corrected")
        uncorrected_df = df[~df["gal_corrected"]]
        corrected_df = df[df["gal_corrected"]]
    else:
        uncorrected_df = copy.deepcopy(df)
        corrected_df = pd.DataFrame()

    extinc = galactic_extinction(uncorrected_df[wavelength_column], **kwargs)
    uncorrected_df[magnitude_column] -= extinc
    uncorrected_df["gal_extinction"] = extinc
    uncorrected_df["gal_corrected"] = True
    return pd.concat([corrected_df, uncorrected_df])

def galactic_extinction(wavelength, ra=None, dec=None, model="fitzpatrick99", wavelength_unit='AA', rv=3.1) -> np.ndarray:

    if (ra is not None) and (dec is not None):
        coords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    else:
        from .const import RA, DEC
        coords = SkyCoord(ra=RA, dec=DEC, unit='deg', frame='icrs')

    sfd = SFDQuery()
    ebv = sfd(coords)
    
    av = rv * ebv
    
    wavelengths_aa = u.Quantity(wavelength, wavelength_unit).to(u.AA).value

    if model == "fitzpatrick99":
        extinc = extinction.fitzpatrick99(np.array(wavelengths_aa), av)
    else:
        raise ValueError(f"Invalid model: {model}")

    return extinc

def host_galaxy_extinction_curve(nu_obs, z=None, model="MW", wavelength_unit='AA') -> np.ndarray:

    if z is None:
        from .const import REDSHIFT
        z = REDSHIFT

    if model not in ["MW", "SMC", "LMC"]:
        raise ValueError(f"Invalid model: {model}")

    tab = asc.read(f"{DATA_DIR}/host_galaxy_extinction.csv")

    interp_func = interp1d(tab[f"nu_{model}"], tab[f"eta_{model}"], 
                          bounds_error=False, fill_value="extrapolate")

    nu_rest = nu_obs * (1. + z)
    eta = interp_func(nu_rest)
    return eta

def host_galaxy_extinction(nu_obs, Av, eta=None, **kwargs) -> np.ndarray:

    if eta is None:
        eta = host_galaxy_extinction_curve(nu_obs, **kwargs)
    
    elif len(eta) != len(nu_obs):
        eta = host_galaxy_extinction_curve(nu_obs, **kwargs)

    return np.exp(-eta * Av / 1.086)

def correct_host_galaxy_extinction(df, Av, z, model="MW", magnitude_column="magnitude", wavelength_column="wavelength") -> pd.DataFrame:
    """
    Corrects the magnitude data for host galaxy extinction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the magnitude data with columns for wavelength and magnitude.
    Av : float
        The extinction in magnitudes.
    z : float
        The redshift of the source.
    model : str, optional
        The extinction model to use. Default is "MW".
    magnitude_column : str, optional
        The name of the column containing the magnitude data. Default is "magnitude".
    wavelength_column : str, optional
        The name of the column containing the wavelength data. Default is "wavelength".
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the corrected magnitude data.
    """
    # 1. Safety checks (Same as your galactic function)
    if "host_corrected" in df.columns and df["host_corrected"].all():
        return df
    
    # 2. Split for partial correction
    if "host_corrected" in df.columns and df["host_corrected"].any():
        uncorrected_df = df[~df["host_corrected"]].copy()
        corrected_df = df[df["host_corrected"]]
    else:
        uncorrected_df = df.copy()
        corrected_df = pd.DataFrame()

    # 3. Calculate Extinction (A_lambda)
    from .utils import unit_conversion
    unit = "AA" if wavelength_column == "wavelength" else "nm"
    nu_obs = unit_conversion(uncorrected_df[wavelength_column].to_numpy(), unit, "Hz")
    
    eta = host_galaxy_extinction_curve(nu_obs, z=z, model=model)
    extinc_mag = eta * Av
    
    # 4. Apply Correction (Subtraction)
    # The error column remains untouched because it's a magnitude error
    uncorrected_df[magnitude_column] -= extinc_mag
    
    # 5. Tag and Merge
    uncorrected_df["host_extinction_mag"] = extinc_mag
    uncorrected_df["host_corrected"] = True
    return pd.concat([corrected_df, uncorrected_df])