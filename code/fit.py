from abc import ABC, abstractmethod
import numpy as np
import scipy
import datetime
import ultranest
from ultranest import stepsampler

class Fitter:
    def __init__(self):
        self.fit_statistic = "chisq"
        self.x_data = None
        self.y_data = None
        self.y_data_error = None
        self.param_bounds = None
        self.prior = None

    @abstractmethod
    def model(self, x_data, *params):
        pass
    
    def y_model(self, params):
        return self.model(self.x_data, *params)

    def log_likelihood(self, params, rescale=True):
        if rescale:
            if self.y_data_error is None:
                scale_factor = min(self.y_data)
            else:
                scale_factor = min(self.y_data_error)
        else:
            scale_factor = 1

        y_data = self.y_data / scale_factor
        y_model = self.model(self.x_data, *params) / scale_factor
        if self.y_data_error is not None:
            y_err = self.y_data_error / scale_factor

        if self.fit_statistic == "chisq":
            if self.y_data_error is None:
                return -1*np.sum((y_data - y_model)**2)
            else:
                return -1*np.sum((y_data - y_model)**2 / y_err**2)

        elif self.fit_statistic == "log_rms":
            return -0.5*np.sum(np.abs((np.log10(self.y_data) - np.log10(self.model(self.x_data, *params)))))
        else:
            raise ValueError(f"Method {self.fit_statistic} not supported")

    def prior_transform(self, cube):
        """Transform the unit cube to the parameter space"""
        params_transformed = np.zeros_like(cube)
        for i, (param, bounds) in enumerate( self.param_bounds.items()):
            if self.prior[param] == "uniform":
                params_transformed[i] = cube[i] * (bounds[1] - bounds[0]) + bounds[0]
            elif self.prior[param] == "log_uniform":
                params_transformed[i] = 10**(cube[i] * (np.log10(bounds[1]) - np.log10(bounds[0])) + np.log10(bounds[0]))
            elif self.prior[param] == "norm":
                params_transformed[i] = scipy.stats.norm.ppf(cube[i], loc=bounds[0], scale=bounds[1])
            else:
                raise ValueError(f"Prior {self.prior[param]} not supported")
        return params_transformed
        
    def run(self, log_dir=None, num_live_points=300, nsteps=None):
        # Create UltraNest sampler
        if self.model is None:
            raise ValueError("Model not set, Analyzer.model must be set first")

        if log_dir == True:
            log_dir = f"ultranest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if nsteps is None:
            nsteps = 2 * len(self.params)

        self.sampler = ultranest.ReactiveNestedSampler(
            self.params,
            self.log_likelihood,
            self.prior_transform,
            log_dir=log_dir,)

        self.sampler.stepsampler = stepsampler.SliceSampler(
            nsteps=nsteps,
            generate_direction=stepsampler.generate_mixture_random_direction,
        )

        # run again:
        self.results = self.sampler.run(
                min_num_live_points=num_live_points,
            )
        return self.sampler