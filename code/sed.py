import matplotlib.pyplot as plt


def plot_sed(df, xdata = "wavelength", ydata = "magnitude", xerr = "filter_width", yerr = "mag_error"):   
    ax = plt.gca()
    ax.errorbar(df[xdata], df[ydata], yerr=df[yerr], xerr=df[xerr], ls="")
    ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=14)
    ax.set_ylabel("Magnitude", fontsize=14)
    ax.invert_yaxis()
    ax.grid()
    plt.show()
