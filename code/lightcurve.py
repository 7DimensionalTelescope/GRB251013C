import matplotlib.pyplot as plt
import numpy as np

def plot_lightcurve(df, label=None, ls="", **kwargs):

    ax = kwargs.pop("ax", None)
    if ax is None:
        ax = plt.gca()
        invert_yaxis = True
    else:
        invert_yaxis = False
    
    data = []
    for idx, row in df.iterrows():
        try:
            data.append([float(row["T-T0"]), float(row["Mag"]), float(row["Error"]), False])
        except:
            continue
            #data.append([float(row["T-T0"]), row["Mag"], row["Error"], True])
    
    data = np.asarray(data)
    upper_limits = data[:,3].astype(bool)
    
    ax.errorbar(data[:,0][~upper_limits], data[:,1][~upper_limits], yerr=data[:,2][~upper_limits], label=label, ls=ls, **kwargs)

    ax.set_xlabel("Time since trigger (hours)")
    ax.set_ylabel("Magnitude")
    ax.set_xscale("log")
    ax.legend()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.grid(which="major", axis="both", linestyle="-", color="gray", alpha=0.8, lw=0.1)
    ax.grid(which="minor", axis="both", linestyle="--", color="gray", alpha=0.5, lw=0.1)

    return ax
