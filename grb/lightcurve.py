import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from .utils import mag_to_flux_mJy

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
