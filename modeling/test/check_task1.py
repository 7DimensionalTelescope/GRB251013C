"""Task 1 verification. Run directly; not a pytest test."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from grb.utils import seconds_from_trigger, model_array
from grb import const

# 1. seconds_from_trigger must not raise
v = seconds_from_trigger("2025-10-14T00:04:00")
assert isinstance(v, float), type(v)
print(f"seconds_from_trigger OK -> {v}")

# 2. it must agree with the baseline implementation in final_model.py
from datetime import datetime
expected = (datetime.strptime("2025-10-14T00:04:00", "%Y-%m-%dT%H:%M:%S")
            - const.TRIGGER_TIME).total_seconds()
assert v == expected, (v, expected)

# 3. microsecond format also works
assert isinstance(seconds_from_trigger("2025-10-14T00:04:00.500"), float)

# 4. bad format still raises ValueError
try:
    seconds_from_trigger("not-a-date"); raise AssertionError("should have raised")
except ValueError:
    pass

# 5. constants present with exact values
assert const.XRT_NU_LO == 7.25e16
assert const.XRT_NU_HI == 2.42e18
assert const.SI_FLARE_FRAC_MAX == 0.5
assert const.HOST_AV_LOG10_MEAN == -0.82
assert const.HOST_AV_LOG10_SIGMA == 0.41
assert const.C_CM_PER_S == 2.99792458e10
assert const.LN10_OVER_2P5 == 0.4 * np.log(10.0)
assert const.XRT_EXCLUDE_TIME_RANGE == (3e3, 1e4)
assert const.XRT_FLARE_START_TIME == 3e3
assert const.XRT_FLARE_END_TIME == 1e4
assert const.MODEL_RESOLUTIONS == (0.1, 0.25, 10)

from VegasAfterglow.units import keV
assert const.XRT_BAND == (0.3 * keV, 10.0 * keV)
assert const.FIT_RESULTS_DIR.name == "fit_results"
assert const.FIT_RESULTS_DIR.parent.name == "modeling"

# 6. model_array unwraps .total and sums .sync + .ssc
class _SS:
    sync = np.array([1.0, 2.0]); ssc = np.array([0.5, 0.5])
class _T:
    total = _SS()
np.testing.assert_array_equal(model_array(_T()), np.array([1.5, 2.5]))
np.testing.assert_array_equal(model_array(np.array([[3.0, 4.0]])), np.array([3.0, 4.0]))
print("Task 1 checks PASSED")
