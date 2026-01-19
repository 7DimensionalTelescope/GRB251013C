import copy
import matplotlib.pyplot as plt
import numpy as np

from ultranest.plot import cornerplot
from .fit import Fitter

from . import prior as prior_module

import pandas as pd

class Analyzer(Fitter):
    def __init__(self, x_data, y_data, model=None, param_bounds=None, prior=None, y_data_error=None):
        super().__init__()

        if isinstance(x_data, pd.Series):
            x_data = x_data.to_numpy("float")
        if isinstance(y_data, pd.Series):
            y_data = y_data.to_numpy("float")

        self.x_data = x_data
        self.y_data = y_data
        self.model = model
        
        if param_bounds is None or prior is None:
            default_param_bounds, default_prior = getattr(prior_module, model.__name__ + "_prior")()
            self.param_bounds = param_bounds or default_param_bounds
            self.prior = prior or default_prior
        else:
            self.param_bounds = param_bounds
            self.prior = prior

        self.params = list(self.param_bounds.keys())
        self.y_data_error = y_data_error

    @property
    def best_fit_params(self):
        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")
        
        return self.results["maximum_likelihood"]["point"]

    @property
    def x_model(self):
        return np.geomspace(min(self.x_data), max(self.x_data), 100)
    
    @property
    def y_model(self):
        return self.model(self.x_model, *self.best_fit_params)


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

    def plot_model(self, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        x_label = kwargs.pop("xlabel", "Time")
        y_label = kwargs.pop("ylabel", "Flux")

        if kwargs.pop("params", None) is not None:
            params = kwargs.pop("params")
            y_model = self.model(self.x_model, *params)
        else:
            y_model = self.y_model

        ax.plot(self.x_model, y_model, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        self._model_params = self.best_fit_params
        return ax

    def plot_corner(self, **kwargs):
        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")

        transformed_samples = copy.deepcopy(self.results)
        for i, (key, prior) in enumerate(self.prior.items()):
            if prior == "log_uniform":
                transformed_samples["weighted_samples"]["points"][:, i] = np.log10(transformed_samples["weighted_samples"]["points"][:, i])
                transformed_samples['paramnames'][i] = fr"log$_{{10}}$({key})"
            else:
                pass

        return cornerplot(transformed_samples, **kwargs)
