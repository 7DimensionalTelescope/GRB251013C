from abc import ABC, abstractmethod
import numpy as np
import scipy
import datetime
import re

class _BilbyFitterLikelihood:
    """bilby.Likelihood-compatible wrapper around Fitter.log_likelihood."""

    def __new__(cls, fitter, bilby_module, param_names, rescale=True):
        class BilbyLikelihood(bilby_module.Likelihood):
            def __init__(self):
                # Keep a parameter dict for bilby v2 compatibility while also
                # accepting explicit parameters for newer interfaces.
                super().__init__(parameters={name: None for name in param_names})

            def log_likelihood(self, parameters=None):
                if parameters is None:
                    parameters = self.parameters
                params = [parameters[name] for name in param_names]
                return fitter.log_likelihood(params, rescale=rescale)

        return BilbyLikelihood()

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

    def log_prior(self, params):
        """Log prior probability for MCMC methods"""
        param_list = list(self.param_bounds.keys())
        log_prob = 0.0
        for i, param_name in enumerate(param_list):
            bounds = self.param_bounds[param_name]
            prior_type = self.prior[param_name]
            val = params[i]
            
            if prior_type == "uniform":
                if bounds[0] <= val <= bounds[1]:
                    log_prob += -np.log(bounds[1] - bounds[0])
                else:
                    return -np.inf
            elif prior_type == "log_uniform":
                if bounds[0] <= val <= bounds[1]:
                    log_prob += -np.log(val) - np.log(np.log(bounds[1]) - np.log(bounds[0]))
                else:
                    return -np.inf
            elif prior_type == "norm":
                mean, std = bounds
                log_prob += scipy.stats.norm.logpdf(val, loc=mean, scale=std)
            elif prior_type == "log_norm":
                mean, std = bounds
                if val > 0:
                    log_prob += scipy.stats.norm.logpdf(np.log10(val), loc=mean, scale=std) - np.log(val * np.log(10))
                else:
                    return -np.inf
            else:
                raise ValueError(f"Prior {prior_type} not supported")
        return log_prob

    def log_probability(self, params):
        """Combined log prior + log likelihood for MCMC"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)

    def prior_transform(self, cube):
        """Transform unit cube to parameter space for nested sampling"""
        params_transformed = np.zeros_like(cube)
        param_list = list(self.param_bounds.keys())
        for i, param_name in enumerate(param_list):
            bounds = self.param_bounds[param_name]
            prior_type = self.prior[param_name]
            
            if prior_type == "uniform":
                params_transformed[i] = cube[i] * (bounds[1] - bounds[0]) + bounds[0]
            elif prior_type == "log_uniform":
                params_transformed[i] = 10**(cube[i] * (np.log10(bounds[1]) - np.log10(bounds[0])) + np.log10(bounds[0]))
            elif prior_type == "norm":
                params_transformed[i] = scipy.stats.norm.ppf(cube[i], loc=bounds[0], scale=bounds[1])
            elif prior_type == "log_norm":
                param_log = scipy.stats.norm.ppf(cube[i], loc=bounds[0], scale=bounds[1])
                params_transformed[i] = 10**param_log
            else:
                raise ValueError(f"Prior {prior_type} not supported")
        return params_transformed
        
    def run(self, method="bilby", **kwargs):
        if method == "emcee":
            self._run_emcee(**kwargs)
        elif method == "ultranest":
            self._run_ultranest(**kwargs)
        elif method == "bilby":
            self._run_bilby(**kwargs)
        elif method == "curve_fit" or method=="quick":
            self._run_curve_fit(**kwargs)
        else:
            raise ValueError(f"Method {method} not supported")

    def _run_curve_fit(self, sigma_range=3.0, **kwargs):
        from scipy.optimize import curve_fit

        param_list = list(self.param_bounds.keys())
        
        # Convert prior-based bounds to curve_fit format: ([lower], [upper])
        lower_bounds = []
        upper_bounds = []
        p0 = []
        
        for param_name in param_list:
            bounds = self.param_bounds[param_name]
            prior_type = self.prior[param_name]
            
            if prior_type == "uniform":
                lower_bounds.append(bounds[0])
                upper_bounds.append(bounds[1])
                p0.append((bounds[0] + bounds[1]) / 2)
            elif prior_type == "log_uniform":
                # For log_uniform: bounds are already (min, max) in linear space
                lower_bounds.append(bounds[0])
                upper_bounds.append(bounds[1])
                # Initial guess: geometric mean
                p0.append(np.sqrt(bounds[0] * bounds[1]))
            elif prior_type == "norm":
                # bounds = (mean, std), convert to (min, max)
                mean, std = bounds
                lower_bounds.append(mean - sigma_range * std)
                upper_bounds.append(mean + sigma_range * std)
                p0.append(mean)
            elif prior_type == "log_norm":
                # bounds = (log_mean, log_std), convert to linear (min, max)
                log_mean, log_std = bounds
                lower_bounds.append(10 ** (log_mean - sigma_range * log_std))
                upper_bounds.append(10 ** (log_mean + sigma_range * log_std))
                p0.append(10 ** log_mean)
            else:
                raise ValueError(f"Prior {prior_type} not supported for curve_fit")
        
        # curve_fit expects sigma for y errors
        sigma = kwargs.pop('sigma', self.y_data_error)
        
        popt, pcov = curve_fit(
            self.model, 
            self.x_data, 
            self.y_data,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            sigma=sigma,
            **kwargs
        )
        
        # Store results in consistent format
        self.results = {
            "maximum_likelihood": {"point": popt},
            "covariance": pcov,
            "paramnames": param_list,
            
        }

        if self.y_data_error is not None:
            self.results["chisq"] = np.sum((self.y_data - self.model(self.x_data, *popt))**2 / self.y_data_error**2)
        
        return self.results

    def _run_emcee(self, nwalkers=32, max_n=100000, check_every=100, tau_threshold=50, tolerance=0.01, p0=None, **kwargs):
        import emcee

        param_list = list(self.param_bounds.keys())
        ndim = len(param_list)
        
        if p0 is None:
            p0 = []
            for _ in range(nwalkers):
                sample = []
                for param_name in param_list:
                    bounds = self.param_bounds[param_name]
                    prior_type = self.prior[param_name]
                    if prior_type == "uniform":
                        sample.append(np.random.uniform(bounds[0], bounds[1]))
                    elif prior_type == "log_uniform":
                        sample.append(10**np.random.uniform(np.log10(bounds[0]), np.log10(bounds[1])))
                    elif prior_type == "norm":
                        sample.append(np.random.normal(bounds[0], bounds[1]))
                    elif prior_type == "log_norm":
                        sample.append(10**np.random.normal(bounds[0], bounds[1]))
                p0.append(sample)
            p0 = np.array(p0)

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)

        old_tau = np.inf
        converged = False
        
        print(f"Running emcee: max_n={max_n}, check_every={check_every}")
        print(f"Convergence criteria: N > {tau_threshold} * tau  AND  d(tau) < {tolerance}")

        # Turn OFF default progress bar (progress=False) so we can print our own status without conflict
        for sample in self.sampler.sample(p0, iterations=max_n, progress=False):
            
            # Check convergence every `check_every` steps
            if self.sampler.iteration % check_every:
                continue

            # Calculate tau
            # tol=0 ensures we get a number even if the chain is short
            tau = self.sampler.get_autocorr_time(tol=0)
            mean_tau = np.mean(tau)

            # Check 1: Chain length vs Tau
            # We need N > threshold * tau
            target_n = tau_threshold * mean_tau
            is_long_enough = self.sampler.iteration > target_n

            # Check 2: Stability of Tau
            change = np.abs(old_tau - mean_tau) / mean_tau
            is_stable = change < tolerance

            # --- PRINT STATUS ---
            # \r overwrites the line. 
            # We format the numbers for readability.
            status = f"Iter: {self.sampler.iteration:5d} | Mean tau: {mean_tau:6.1f} | Target N: {target_n:6.0f} | d(tau): {change:.3f}"
            print(status, end="\r")

            # Update old_tau only if we have a valid measurement
            if not np.isnan(mean_tau):
                old_tau = mean_tau

            # Convergence Logic
            if np.any(np.isnan(tau)) or not is_long_enough:
                continue

            if is_stable:
                converged = True
                print(f"\nConverged! Stopping at iteration {self.sampler.iteration}")
                break
        
        if not converged:
            print(f"\nWarning: Reached max_n ({max_n}) without fully converging.")
            try:
                mean_tau = np.mean(self.sampler.get_autocorr_time(tol=0))
            except:
                # If calculation fails, assume roughly 1/50th of the chain
                mean_tau = self.sampler.iteration / tau_threshold

        # Post-processing (Burn-in & Thinning)
        burnin = int(2 * mean_tau)
        thin = int(max(1, mean_tau / 2))

        print(f"Post-processing: Discarding {burnin}, Thinning by {thin}")
        
        try:
            flat_samples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
            flat_log_prob = self.sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            
            max_idx = np.argmax(flat_log_prob)
            
            self.results = {
                "samples": flat_samples,
                "maximum_likelihood": {"point": flat_samples[max_idx], "log_prob": flat_log_prob[max_idx]},
                "weighted_samples": {"points": flat_samples}, # emcee samples are equally weighted
                "paramnames": param_list,
                "tau": mean_tau
            }
        except Exception as e:
            print(f"Error processing chains (likely too short): {e}")
            self.results = None

        return self.sampler

    def _run_ultranest(self, log_dir=None, num_live_points=300, nsteps=None, **kwargs):
        import ultranest
        from ultranest import stepsampler

        param_list = list(self.param_bounds.keys())
        
        self.sampler = ultranest.ReactiveNestedSampler(
            param_list,
            self.log_likelihood,
            self.prior_transform,
            log_dir=log_dir)

        if nsteps is None:
            nsteps = 2 * len(param_list)

        self.sampler.stepsampler = stepsampler.SliceSampler(
            nsteps=nsteps,
            generate_direction=stepsampler.generate_mixture_random_direction)

        self.results = self.sampler.run(min_num_live_points=num_live_points)
        return self.sampler

    def _build_bilby_priors(self):
        import bilby

        param_names = getattr(self, "params", list(self.param_bounds.keys()))
        priors = bilby.core.prior.PriorDict()
        for name in param_names:
            bounds = self.param_bounds[name]
            prior_type = self.prior[name]

            if prior_type == "uniform":
                priors[name] = bilby.core.prior.Uniform(bounds[0], bounds[1], name=name)
            elif prior_type == "log_uniform":
                priors[name] = bilby.core.prior.LogUniform(bounds[0], bounds[1], name=name)
            elif prior_type == "norm":
                priors[name] = bilby.core.prior.Gaussian(mu=bounds[0], sigma=bounds[1], name=name)
            elif prior_type == "log_norm":
                # Existing convention in this project: log10(param) ~ Normal(mean, std).
                # bilby LogNormal is defined in natural-log space.
                ln10 = np.log(10.0)
                
                priors[name] = bilby.core.prior.LogNormal(
                    mu=bounds[0] * ln10,
                    sigma=bounds[1] * ln10,
                    name=name,
                )
            else:
                raise ValueError(f"Prior {prior_type} not supported")

        return priors

    # def _build_bilby_likelihood(self):
        # class BilbyLikelihood(bilby.Likelihood):
        #     def __init__(self, x_data, y_data, y_data_error):
        #         super().__init__(parameters=self.param_bounds.keys())

        #     def log_likelihood(self):
        #         return self.log_likelihood(self.params)


    def _run_bilby(self, sampler="dynesty", label="grb_fit", outdir="bilby_out", rescale=True, nlive=800, dlogz=0.2, walks=20, resume=True, **kwargs):
        import bilby

        if self.y_data_error is None:
            raise ValueError("bilby requires y_data_error for GaussianLikelihood")

        param_names = getattr(self, "params", list(self.param_bounds.keys()))
        priors = self._build_bilby_priors()
        likelihood = _BilbyFitterLikelihood(self, bilby, param_names, rescale=rescale)

        self.sampler = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sampler,
            label=label,
            outdir=outdir,
            maxiter=None,
            maxcall=None,
            nlive = nlive,
            dlogz=dlogz,
            walks=walks,
            resume=resume,
            **kwargs,
        )

        posterior = self.sampler.posterior
        samples = posterior[param_names].to_numpy()

        if "log_likelihood" in posterior.columns:
            log_like = posterior["log_likelihood"].to_numpy()
            best_idx = int(np.argmax(log_like))
            best_log_like = float(log_like[best_idx])
        else:
            best_idx = int(np.argmax(self.sampler.log_likelihood_evaluations))
            best_log_like = float(self.sampler.log_likelihood_evaluations[best_idx])

        # bilby posterior samples are already posterior-distributed, so
        # represent them as equally weighted samples in the UltraNest-like schema.
        nsamples = len(samples)
        if nsamples > 0:
            weights = np.full(nsamples, 1.0 / nsamples)
        else:
            weights = np.array([])

        self.results = {
            "samples": samples,
            "maximum_likelihood": {"point": samples[best_idx], "log_prob": best_log_like},
            "weighted_samples": {"points": samples, "weights": weights},
            "paramnames": param_names,
        }
        return self.sampler


    def print_posterior(self):
        from IPython.display import display, Math

        for i in range(len(self.params)):
            mcmc = np.percentile(self.results["samples"][:, i], [16, 50, 84])
            q = np.diff(mcmc)

            name = self.params[i]

            if "_" in name:
                # Finds underscore followed by text and wraps the text in {}
                latex_name = re.sub(r"_([a-zA-Z0-9]+)", r"_{\1}", name)
            else:
                latex_name = name

            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], latex_name)
            display(Math(txt))