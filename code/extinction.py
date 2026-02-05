from pathlib import Path
from astropy.io import ascii as asc
from scipy.interpolate import interp1d
from .const import DATA_DIR

def galactic_extinction(bandpass, system = ["SDSS", "Landolt", "UKIRT", "PS1"]):
    tab = asc.read(f"{DATA_DIR}/extinction_calculator.csv")
    tab = tab[tab['Refcode of the publications']=="2011ApJ...737..103S"]
    
    if bandpass == "Ks":
        bandpass = "K"
    elif bandpass == "Y":
        bandpass = "y"
    
    band = [str(sys) + " " + str(bandpass) for sys in system]
    unique_band = [tab[tab["Bandpass"] == b] for b in band if b in tab["Bandpass"]]
    
    if len(unique_band) >= 1:
        return unique_band[0]
    else:
        return None

def galactic_extinction_wavelength(wavelength, wavelength_unit='micron', E_BV=1.0):
    """
    Get Galactic extinction for a specific wavelength using Schlafly & Finkbeiner (2011)
    
    Args:
        wavelength: wavelength value (can be scalar or array)
        wavelength_unit: unit of wavelength ('micron', 'nm', 'angstrom')
        E_BV: color excess E(B-V) (default 1.0, returns A_λ/E(B-V))
    
    Returns:
        A_λ or A_λ/E(B-V) depending on E_BV value
    """
    tab = asc.read(f"{DATA_DIR}/extinction_calculator.csv")
    tab = tab[tab['Refcode of the publications']=="2011ApJ...737..103S"]
    
    # Convert wavelength to microns for interpolation
    if wavelength_unit == 'nm':
        wavelength_micron = wavelength / 1000.0
    elif wavelength_unit == 'angstrom' or wavelength_unit == 'angstroms':
        wavelength_micron = wavelength / 10000.0
    elif wavelength_unit == 'micron' or wavelength_unit == 'microns':
        wavelength_micron = wavelength
    else:
        raise ValueError(f"Unknown wavelength_unit: {wavelength_unit}")
    
    # Get wavelength and extinction data
    wl_data = tab['Central Wavelength'].data  # in microns
    A_data = tab['Galactic Extinction'].data   # A_λ/E(B-V) or similar
    
    # Interpolate extinction for requested wavelength(s)
    interp_func = interp1d(wl_data, A_data, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    
    A_lambda_over_EBV = interp_func(wavelength_micron)
    
    # Return A_λ = (A_λ/E(B-V)) * E(B-V)
    return A_lambda_over_EBV * E_BV

def host_galaxy_extinction(model, nu=None, z=None, show_plot=False, **kwargs):
    tab = asc.read(f"{DATA_DIR}/host_galaxy_extinction.csv")
    if show_plot:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        ax.plot(tab[f"nu_{model}"], tab[f"eta_{model}"], **kwargs)
        ax.set_xlabel("Rest frequency [Hz]", fontsize=13)
        ax.set_ylabel(r"$\eta$($\nu$)", fontsize=13)
        ax.set_xscale("log")

    else:
        eta_model = interp1d(tab[f"nu_{model}"], tab[f"eta_{model}"])
        if (nu is not None) and (z is not None):
            # nu is observed frequency, convert to rest-frame: nu_rest = nu_obs * (1+z)
            # The table has rest-frame frequencies
            return eta_model(nu * (1. + z))
        elif nu is not None:
            # If z not provided, assume nu is already rest-frame
            return eta_model(nu)
        else:
            return eta_model
