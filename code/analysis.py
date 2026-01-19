import copy
import matplotlib.pyplot as plt
import numpy as np

from ultranest.plot import cornerplot
from .fit import Fitter

class Analyzer(Fitter):
    def __init__(self, x_data, y_data, model, param_bounds, prior, y_data_error=None):
        super().__init__()

        self.x_data = x_data
        self.y_data = y_data
        self.y_data_error = y_data_error
        self.model = model
        self.param_bounds = param_bounds
        self.prior = prior
        self.params = list(self.param_bounds.keys())

    
    @property
    def best_fit_params(self):
        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")
        
        return self.results["maximum_likelihood"]["point"]

    def plot_data(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        x_label = kwargs.pop("xlabel", "Time")
        y_label = kwargs.pop("ylabel", "Flux")

        if self.y_data_error is not None:
            ax.errorbar(self.x_data, self.y_data, yerr=self.y_data_error, ls="", **kwargs)
        else:
            ax.plot(self.x_data, self.y_data, marker=kwargs.pop("marker", "o"), ls="", **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_model(self, params=None, ax=None, **kwargs):

        if params is None:
            params = self.best_fit_params

        if ax is None:
            ax = plt.gca()

        x_label = kwargs.pop("xlabel", "Time")
        y_label = kwargs.pop("ylabel", "Flux")

        ax.plot(self.x_data, self.model(self.x_data, *params), **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_corner(self, **kwargs):
        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")

        transformed_samples = copy.deepcopy(self.results)
        for i, (key, prior) in enumerate(self.prior.items()):
            if prior == "log_uniform":
                transformed_samples["weighted_samples"]["points"][:, i] = np.log10(transformed_samples["weighted_samples"]["points"][:, i])
                transformed_samples['paramnames'][i] = fr"log$_{{10}}$({key})"
            elif prior == "uniform":
                pass
            else:
                raise ValueError(f"Prior {prior} not supported")

        return cornerplot(transformed_samples, **kwargs)
