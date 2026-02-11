import numpy as np
import matplotlib.pyplot as plt
from .utils import mJy_to_erg_cm2_s, unit_conversion
from .functions import power_law

def plot_sed_from_samples(dataset, ax=None, x_unit="Hz", y_unit="mJy", add_median=False, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax = plot_band(dataset[:,0], dataset[:,1], dataset[:,3], ax=ax, x_unit=x_unit, y_unit=y_unit, **kwargs)
    
    if add_median:
        plot_sed(dataset[:,0], dataset[:, 2], ax=ax, x_unit=x_unit, y_unit=y_unit, **kwargs)

    return ax

def convert_units(nu, flux, x_unit="Hz", y_unit="mJy"):
    if x_unit == "eV":
        x_data = unit_conversion(nu, "Hz", "eV")
    elif x_unit == "AA":
        x_data = unit_conversion(nu, "Hz", "AA")
        
    if y_unit == "sed":
        y_data = mJy_to_erg_cm2_s(flux, nu)
        y_unit = r"erg/cm$^2$/s"
    
    return x_data, y_data

def _sed_plot_setup(x_unit, y_unit, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")
    if x_unit == "eV":
        x_label = r"Energy ($eV$)"
    elif x_unit == "AA":
        x_label = r"Wavelength ($\AA$)"
    else:
        x_label = f"Frequency ({x_unit})"

    ax.set_xlabel(x_label, fontsize=14)

    if y_unit == "sed":
        y_label = r"Energy Flux ($erg/cm^2/s$)"
    else:
        y_label = f"Flux ({y_unit})"

    ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(labelsize=12)
    return ax


def plot_sed(nu, flux, ax=None, x_unit="Hz", y_unit="mJy", **kwargs):
    if ax is None:
        ax = plt.gca()

    x_data, y_data = convert_units(nu, flux, x_unit, y_unit)
    
    ax.plot(x_data, y_data, **kwargs)

    _sed_plot_setup(x_unit, y_unit, ax)

    return ax

def plot_band(nu, flux_l, flux_h, ax=None, x_unit="Hz", y_unit="mJy", alpha=0.5, **kwargs):
    if ax is None:
        ax = plt.gca()

    x_data, y_data = convert_units(nu, flux_l, x_unit, y_unit)
    x_data_h, y_data_h = convert_units(nu, flux_h, x_unit, y_unit)

    ax.fill_between(x_data, y_data, y_data_h, alpha=alpha, **kwargs)

    _sed_plot_setup(x_unit, y_unit, ax)

    return ax
    

