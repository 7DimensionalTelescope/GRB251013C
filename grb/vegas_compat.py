"""Thread-safety patch for VegasAfterglow's bilby fitting path.

VegasAfterglow 2.0.6 runs bilby samplers (dynesty, ...) with a thread pool,
but bilby sets ``likelihood.parameters`` and then calls ``log_likelihood()``
on a single shared AfterglowLikelihood object from every pool thread. With
npool > 1 the threads overwrite each other's parameters mid-evaluation, so
dynesty sees non-reproducible likelihoods and its slice sampler aborts with
"Slice sampler has failed to find a valid point".

Importing this module replaces AfterglowLikelihood with a subclass whose
``parameters`` dict is thread-local, making the set-then-evaluate sequence
race-free. Remove once fixed upstream (VegasAfterglow > 2.0.6).
"""
import threading

import numpy as np

from VegasAfterglow.fitting import samplers as _samplers
from VegasAfterglow.fitting import utils as _vutils

_BaseLikelihood = _vutils.AfterglowLikelihood


class ThreadSafeAfterglowLikelihood(_BaseLikelihood):
    """AfterglowLikelihood with thread-local parameters for thread-pool sampling."""

    __slots__ = ("_tls",)

    _STATE_KEYS = ("parameters", "param_keys", "_fitter", "_theta",
                   "_transformer", "_log_likelihood_fn")

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_tls", threading.local())
        super().__init__(*args, **kwargs)

    @property
    def parameters(self):
        tls = self._tls
        if not hasattr(tls, "parameters"):
            tls.parameters = {key: None for key in self.param_keys}
        return tls.parameters

    @parameters.setter
    def parameters(self, value):
        self._tls.parameters = dict(value) if value else {}

    def __getstate__(self):
        return {k: getattr(self, k) for k in self._STATE_KEYS}

    def __setstate__(self, state):
        object.__setattr__(self, "_tls", threading.local())
        for k, v in state.items():
            setattr(self, k, v)

    def log_likelihood(self, parameters=None) -> float:
        p = self.parameters
        if parameters is not None:
            p.update(parameters)
        theta = np.array([p[key] for key in self.param_keys], dtype=np.float64)

        try:
            params = self._transformer(theta)
            chi2 = self._fitter._evaluate(params)
            if not np.isfinite(chi2):
                return -np.inf
            return self._log_likelihood_fn(chi2)
        except Exception:
            return -np.inf


_vutils.AfterglowLikelihood = ThreadSafeAfterglowLikelihood
_samplers.AfterglowLikelihood = ThreadSafeAfterglowLikelihood
