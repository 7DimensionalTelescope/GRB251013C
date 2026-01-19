from abc import ABC, abstractmethod
import numpy as np
import scipy
import datetime
import ultranest
from ultranest import stepsampler

class Fitter:
    def __init__(self):
        self.result = None
        self.fit_statistic = "chisq"

    @abstractmethod
    def y_data(self):
        pass

    @abstractmethod
    def x_data(self):
        pass
    
    @abstractmethod
    def model(self, x_data, *params):
        pass
    
    @abstractmethod
    def param_bounds(self):
        pass

    @abstractmethod
    def prior(self):
        return 0

    

    def y_model(self, params):
        return self.model(self.x_data, *params)

    def log_likelihood(self, params, rescale=True):
        if rescale:
            y_data = self.y_data / min(self.y_data_error)
            y_model = self.model(self.x_data, *params) / min(self.y_data_error)
            y_err = self.y_data_error / min(self.y_data_error)
        else:
            y_data = self.y_data
            y_model = self.model(self.x_data, *params)
            y_err = self.y_data_error

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
            elif self.prior[param] == "normal":
                params_transformed[i] = cube[i] * 10**scipy.stats.norm.ppf(cube[i], bounds[0], bounds[1])
            else:
                raise ValueError(f"Prior {self.prior[param]} not supported")
        return params_transformed
        
    def run(self, log_dir=None, num_live_points=300, nsteps=None):
        # Create UltraNest sampler
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