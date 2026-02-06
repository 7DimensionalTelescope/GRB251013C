import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fit import Fitter

class Analyzer(Fitter):

    def __init__(self, x_data, y_data, model=None, param_bounds=None, prior=None, x_data_error=None, y_data_error=None, data_type="lightcurve"):
        super().__init__()

        if isinstance(x_data, pd.Series):
            x_data = x_data.to_numpy("float")
        if isinstance(y_data, pd.Series):
            y_data = y_data.to_numpy("float")

        self.x_data = x_data
        self.y_data = y_data
        self.x_data_error = x_data_error
        self.y_data_error = y_data_error
        
        self.data_type = data_type
        self.model = model

        self.set_params(param_bounds, prior)
        
        if self.data_type == "lightcurve":
            self._x_damper = 0.2
        elif self.data_type == "sed":
            self._x_damper = 0.05
        
    def set_params(self, param_bounds=None, prior=None):

        if param_bounds is None or prior is None:
            from . import prior as prior_module

            default_param_bounds, default_prior = getattr(prior_module, self.model.__name__ + "_prior")(data_type=self.data_type)
            self.param_bounds = param_bounds or default_param_bounds
            self.prior = prior or default_prior
        else:
            self.param_bounds = param_bounds
            self.prior = prior

        self.params = list(self.param_bounds.keys())

    @property
    def best_fit_params(self):
        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")
        
        return self.results["maximum_likelihood"]["point"]

    @property
    def x_model(self):
        return np.geomspace(min(self.x_data)*(1-self._x_damper), max(self.x_data)*(1+self._x_damper), 100)
    
    @property
    def y_model(self):
        return self.model(self.x_model, *self.best_fit_params)

    def plot_data(self, ax=None, scale=1, **kwargs):
        if ax is None:
            ax = plt.gca()

        x_label = kwargs.pop("xlabel", self._default_x_label)
        y_label = kwargs.pop("ylabel", self._default_y_label)

        if self.y_data_error is not None:
            ax.errorbar(self.x_data, self.y_data*scale, yerr=self.y_data_error*scale, xerr=self.x_data_error, ls="", **kwargs)
        else:
            ax.plot(self.x_data, self.y_data*scale, marker=kwargs.pop("marker", "o"), ls="", **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_model(self, ax=None, scale=1, show_param = False,**kwargs):

        if ax is None:
            ax = plt.gca()

        x_label = kwargs.pop("xlabel", self._default_x_label)
        y_label = kwargs.pop("ylabel", self._default_y_label)

        if kwargs.pop("params", None) is not None:
            params = kwargs.pop("params")
            y_model = self.model(self.x_model, *params)
        else:
            y_model = self.y_model

        ax.plot(self.x_model, y_model*scale, **kwargs)

        if show_param:
            loc_idx=0
            for i, param in enumerate(self.params):
                if param.startswith("F"):
                    continue
                ax.text(0.80, 0.95-loc_idx*0.05, f"{param}: {self.best_fit_params[i]:.1f}", transform=ax.transAxes, ha="left", va="top")
                loc_idx += 1

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        self._model_params = self.best_fit_params
        return ax

    def plot_corner(self, **kwargs):
        from ultranest.plot import cornerplot

        if self.results is None:
            raise ValueError("No results found, Analyzer.run() must be called first")

        transformed_samples = copy.deepcopy(self.results)
        for i, (key, prior) in enumerate(self.prior.items()):
            if prior == "log_uniform":
                transformed_samples["weighted_samples"]["points"][:, i] = np.log10(transformed_samples["weighted_samples"]["points"][:, i])
                transformed_samples['paramnames'][i] = fr"log$_{{10}}$({key})"
            elif prior == "log_norm":
                transformed_samples["weighted_samples"]["points"][:, i] = np.log10(transformed_samples["weighted_samples"]["points"][:, i])
                transformed_samples['paramnames'][i] = fr"log$_{{10}}$({key})"
                pass

        return cornerplot(transformed_samples, **kwargs)

    @property
    def _default_x_label(self):
        if self.data_type == "lightcurve":
            return "Time"
        elif self.data_type == "sed":
            return "Frequency"
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

    @property
    def _default_y_label(self):
        if self.data_type == "lightcurve":
            return "Flux"
        elif self.data_type == "sed":
            return "Flux"
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")
