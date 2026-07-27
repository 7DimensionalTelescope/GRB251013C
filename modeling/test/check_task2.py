"""Task 2 verification. Run directly."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from grb.extinction import host_extinction_attenuation

nu = np.array([3.93e14, 5.0e14])
np.testing.assert_array_equal(host_extinction_attenuation(nu, 0, 1.0), np.ones(2))

from VegasAfterglow.extinction import BUILTIN_LAWS
from grb.const import C_CM_PER_S, LN10_OVER_2P5
a_v, z = 0.0254, 1.0
lam = C_CM_PER_S / nu / (1.0 + z)
expected = np.exp(-a_v * LN10_OVER_2P5 * BUILTIN_LAWS["smc"](lam))
np.testing.assert_array_equal(host_extinction_attenuation(nu, a_v, z), expected)

assert host_extinction_attenuation(nu, a_v, z)[0] == \
       host_extinction_attenuation(nu, a_v, z, profile="smc")[0]
print("Task 2 checks PASSED")
