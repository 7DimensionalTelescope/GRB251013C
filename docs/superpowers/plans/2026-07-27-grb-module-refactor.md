# GRB Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `modeling/final_model.py`, `modeling/final_model_plotting.py`, and `modeling/utils.py` into the `grb` package, replacing them with a thin root-level driver that produces bitwise-identical results.

**Architecture:** `grb/` stays a flat package of single-purpose modules. Each moved symbol lands in the module its name predicts; five new flat modules (`params`, `likelihood`, `spectral_index`, `results`, `plotting`) hold concerns with no existing home. A root `fit_final_model.py` becomes the CLI. Equivalence is proven by importing the baseline modules out of git and asserting exact array equality.

**Tech Stack:** Python 3.13 (conda env `grb251013c`), VegasAfterglow 2.0.6, emcee, numpy, pandas, astropy, matplotlib, corner.

## Global Constraints

- **Interpreter:** always `/home/dtak/miniconda3/envs/grb251013c/bin/python`. Never bare `python`.
- **Worktree:** all work happens in `/data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor`.
- **Baseline commit:** `a40dd204e0b2bc7e5286387548fe14f75328e9f8`. Never amend or rebase past it.
- **No numeric change.** Every value the fit produces must be bitwise identical to baseline. Use `np.testing.assert_array_equal`, never `assert_allclose`.
- **Copy verbatim.** Where a step says "copy verbatim", reproduce the body character-for-character including comments and docstrings. Do not reformat, rename locals, or "improve" anything.
- **`grb/` must never import from `modeling/`.**
- **Config comes from `/home/dtak/research/grb/.env`** (parent of the real repo). Note: the worktree lives deeper in the tree, so `load_dotenv()` still walks up and finds it. Verify early — Task 1 Step 2 depends on it.
- **Commit after every task.** Author `Donggeun Tak <takdg123@gmail.com>`, trailer `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **`modeling/fit_results/` is gitignored** (2.6 GB). Never `git add -f` it.

## Symbol destination map

Authoritative for every task. `modeling/utils.py` = `MU`, `modeling/final_model.py` = `FM`, `modeling/final_model_plotting.py` = `FMP`.

| Symbol | From | To |
|---|---|---|
| `XRT_EXCLUDE_TIME_RANGE`, `XRT_FLARE_START_TIME`, `XRT_FLARE_END_TIME`, `HOST_AV_LOG10_MEAN`, `HOST_AV_LOG10_SIGMA`, `C_CM_PER_S`, `LN10_OVER_2P5` | MU:50-56 | `grb/const.py` |
| `XRT_BAND`, `XRT_NU_LO`, `XRT_NU_HI`, `SI_FLARE_FRAC_MAX`, `FIT_RESULTS_DIR` | FM:46-57 | `grb/const.py` |
| `model_array` | MU:279-284 | `grb/utils.py` |
| `xrt_flux_error` | MU:80-84 | `grb/utils.py` — already exists as `flux_error`; delete copy, rename call sites |
| `seconds_from_trigger` | FM:60-67 | `grb/utils.py` — already exists but broken; fix it |
| `norris_flare` | FM:70-81 | `grb/functions.py` — already exists identical; delete copy |
| `host_extinction_attenuation` | MU:87-94 | `grb/extinction.py` |
| `make_core_model`, `make_wing_model` | FM:135-191 | `grb/modeling.py` — already exist; de-duplicate |
| `load_all_optical_data` | FM:84-132 | `grb/modeling.py` — exists as `load_all_data`; rename |
| `ParamDefWithPrior` | MU:10-47 | `grb/params.py` |
| `make_param_defs` | FM:194-250 | `grb/params.py` |
| `default_nwalkers` | MU:59-60 | `grb/params.py` |
| `load_xrt_spectral_index` | MU:101-133 | `grb/spectral_index.py` |
| `compute_break_frequencies` | MU:136-174 | `grb/spectral_index.py` |
| `compute_p_prior_from_spectral_index` | MU:177-217 | `grb/spectral_index.py` |
| `spectral_index_model`, `spectral_index_chi2` | FM:253-308 | `grb/likelihood.py` |
| `compute_model_flux_all_bands` | FM:311-392 | `grb/likelihood.py` |
| `log_likelihood`, `log_prior`, `log_probability` | FM:395-462 | `grb/likelihood.py` |
| `top_k_samples` | MU:63-77 | `grb/results.py` |
| `read_labels` | MU:97-98 | `grb/results.py` |
| `latest_result_dir` | MU:220-227 | `grb/results.py` |
| `load_best_fit_params` | FMP:34-61 | `grb/results.py` |
| `plot_corner` | MU:230-262 | `grb/plotting.py` |
| `set_log_y_limits` | MU:265-276 | `grb/plotting.py` |
| `compute_model_components` | FMP:64-155 | `grb/plotting.py` |
| `plot_light_curves` | FMP:158-353 | `grb/plotting.py` |
| `plot_spectral_index_comparison` | FMP:356-466 | `grb/plotting.py` |

---

### Task 1: Constants, `grb/utils.py` fixes, and the extinction guard bug

**Files:**
- Modify: `grb/const.py` (append after line 29)
- Modify: `grb/utils.py:110-119` (fix `seconds_from_trigger`), append `model_array`
- Modify: `grb/extinction.py:19`

**Interfaces:**
- Consumes: nothing.
- Produces: `grb.const.XRT_BAND: tuple[float, float]`, `grb.const.XRT_NU_LO: float`, `grb.const.XRT_NU_HI: float`, `grb.const.SI_FLARE_FRAC_MAX: float`, `grb.const.FIT_RESULTS_DIR: pathlib.Path`, `grb.const.HOST_AV_LOG10_MEAN: float`, `grb.const.HOST_AV_LOG10_SIGMA: float`, `grb.const.XRT_EXCLUDE_TIME_RANGE: tuple[float, float]`, `grb.const.XRT_FLARE_START_TIME: float`, `grb.const.XRT_FLARE_END_TIME: float`, `grb.const.C_CM_PER_S: float`, `grb.const.LN10_OVER_2P5: float`; `grb.utils.model_array(model_output) -> np.ndarray`; working `grb.utils.seconds_from_trigger(date_obs) -> float`.

- [ ] **Step 1: Confirm the env loads from inside the worktree**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
/home/dtak/miniconda3/envs/grb251013c/bin/python -c \
  "from grb.const import REDSHIFT, D_L, TRIGGER_TIME; print(REDSHIFT, D_L, TRIGGER_TIME)"
```

Expected: three non-empty values. If this raises `TypeError: float() argument must be...`, `load_dotenv()` did not find `/home/dtak/research/grb/.env` from the worktree path. **Stop and report** — every later task depends on this.

- [ ] **Step 2: Write the failing check for `seconds_from_trigger`**

Create `modeling/test/check_task1.py`:

```python
"""Task 1 verification. Run directly; not a pytest test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
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
```

- [ ] **Step 3: Run it and watch it fail**

```bash
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/check_task1.py
```

Expected: `AttributeError: module 'datetime' has no attribute 'strptime'`.

- [ ] **Step 4: Fix `grb/utils.py:110-119`**

Replace the whole `seconds_from_trigger` function with:

```python
def seconds_from_trigger(date_obs):
    """Convert date_obs string to seconds from trigger"""
    from datetime import datetime
    from .const import TRIGGER_TIME
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.strptime(str(date_obs), fmt) - TRIGGER_TIME).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date_obs format: {date_obs}")
```

- [ ] **Step 5: Append `model_array` to `grb/utils.py`**

```python
def model_array(model_output):
    if hasattr(model_output, "total"):
        model_output = model_output.total
    if hasattr(model_output, "sync"):
        model_output = np.asarray(model_output.sync) + np.asarray(model_output.ssc)
    return np.asarray(model_output).squeeze()
```

- [ ] **Step 6: Append constants to `grb/const.py`** (after `MODEL_RESOLUTIONS` on line 29)

```python
from pathlib import Path
import numpy as np
from VegasAfterglow.units import keV

FIT_RESULTS_DIR = Path(BASE_DIR) / "modeling" / "fit_results"

# XRT 0.3-10 keV band, and the two frequencies (Hz) between which the local
# synchrotron slope is measured for the spectral-index constraint.
XRT_BAND = (0.3 * keV, 10.0 * keV)
XRT_NU_LO = 7.25e16
XRT_NU_HI = 2.42e18

# The spectral-index constraint is applied only where the phenomenological
# flare contributes less than this fraction of the XRT flux.
SI_FLARE_FRAC_MAX = 0.5

XRT_EXCLUDE_TIME_RANGE = (3e3, 1e4)
XRT_FLARE_START_TIME = 3e3
XRT_FLARE_END_TIME = 1e4

# Log10 Gaussian prior on host-galaxy A_V.
HOST_AV_LOG10_MEAN = -0.82
HOST_AV_LOG10_SIGMA = 0.41

C_CM_PER_S = 2.99792458e10
LN10_OVER_2P5 = 0.4 * np.log(10.0)
```

- [ ] **Step 7: Fix the guard bug at `grb/extinction.py:19`**

Change `df["corrected"].all()` to `df["gal_corrected"].all()`. That line only.

- [ ] **Step 8: Run the check — expect PASS**

```bash
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/check_task1.py
```

Expected: `Task 1 checks PASSED`.

- [ ] **Step 9: Commit**

```bash
git add grb/const.py grb/utils.py grb/extinction.py modeling/test/check_task1.py
git commit -m "Add shared constants and model_array to grb; fix two latent bugs

- grb/utils.seconds_from_trigger raised AttributeError (import datetime vs
  from datetime import datetime), so grb.modeling.load_all_data never ran.
- grb/extinction.py guard read a non-existent \"corrected\" column.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `host_extinction_attenuation` into `grb/extinction.py`

**Files:**
- Modify: `grb/extinction.py` (append)
- Create: `modeling/test/check_task2.py`

**Interfaces:**
- Consumes: `grb.const.C_CM_PER_S`, `grb.const.LN10_OVER_2P5`.
- Produces: `grb.extinction.host_extinction_attenuation(nu_hz, a_v, redshift, profile="smc") -> np.ndarray`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task2.py`:

```python
"""Task 2 verification. Run directly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from grb.extinction import host_extinction_attenuation

nu = np.array([3.93e14, 5.0e14])
# a_v == 0 short-circuits to ones
np.testing.assert_array_equal(host_extinction_attenuation(nu, 0, 1.0), np.ones(2))

# non-zero a_v matches the explicit SMC formula
from VegasAfterglow.extinction import BUILTIN_LAWS
from grb.const import C_CM_PER_S, LN10_OVER_2P5
a_v, z = 0.0254, 1.0
lam = C_CM_PER_S / nu / (1.0 + z)
expected = np.exp(-a_v * LN10_OVER_2P5 * BUILTIN_LAWS["smc"](lam))
np.testing.assert_array_equal(host_extinction_attenuation(nu, a_v, z), expected)

# default profile is smc, NOT the MW default used by host_galaxy_extinction
assert host_extinction_attenuation(nu, a_v, z)[0] == \
       host_extinction_attenuation(nu, a_v, z, profile="smc")[0]
print("Task 2 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ImportError: cannot import name 'host_extinction_attenuation'`**

```bash
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/check_task2.py
```

- [ ] **Step 3: Append to `grb/extinction.py`**

```python
from VegasAfterglow.extinction import BUILTIN_LAWS
from .const import C_CM_PER_S, LN10_OVER_2P5


def host_extinction_attenuation(nu_hz, a_v, redshift, profile="smc"):
    """Host-galaxy extinction as a multiplicative attenuation on model flux.

    Uses VegasAfterglow's built-in extinction laws, defaulting to SMC, and is
    what the afterglow fit applies inside the model.

    NOTE: this is NOT the same code path as `host_galaxy_extinction` above,
    which interpolates the tabulated data/host_galaxy_extinction.csv curve and
    defaults to MW. The two disagree; the fit uses this one. See
    docs/superpowers/specs/2026-07-27-grb-module-refactor-design.md.
    """
    if a_v == 0:
        return np.ones_like(np.asarray(nu_hz, dtype=float))

    law = BUILTIN_LAWS[profile]
    lambda_rest_cm = C_CM_PER_S / np.asarray(nu_hz, dtype=float) / (1.0 + redshift)
    k_lambda = law(lambda_rest_cm)
    return np.exp(-a_v * LN10_OVER_2P5 * k_lambda)
```

- [ ] **Step 4: Run the check — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add grb/extinction.py modeling/test/check_task2.py
git commit -m "Move host_extinction_attenuation into grb/extinction.py

Documents that it is distinct from the tabulated host_galaxy_extinction and
that the fit uses this SMC-based path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `grb/spectral_index.py`

**Files:**
- Create: `grb/spectral_index.py`
- Create: `modeling/test/check_task3.py`

**Interfaces:**
- Consumes: `grb.const.DATA_DIR`.
- Produces: `load_xrt_spectral_index(data_dir=None) -> dict` with keys `time, time_low, time_high, beta, beta_err_low, beta_err_high`; `compute_break_frequencies(params, z, t_obs) -> dict` with keys `nu_m, nu_c`; `compute_p_prior_from_spectral_index(xrt_index_data, cooling_regime="slow") -> tuple[float, float]`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task3.py`:

```python
"""Task 3 verification. Compares against the baseline implementation."""
import sys, os, subprocess, tempfile, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

# materialise the baseline modeling/utils.py out of git
tmp = tempfile.mkdtemp()
src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/utils.py"])
open(os.path.join(tmp, "baseline_utils.py"), "wb").write(src)
spec = importlib.util.spec_from_file_location("baseline_utils",
                                              os.path.join(tmp, "baseline_utils.py"))
bu = importlib.util.module_from_spec(spec); spec.loader.exec_module(bu)

from grb import spectral_index as si

old = bu.load_xrt_spectral_index(); new = si.load_xrt_spectral_index()
assert set(old) == set(new), (set(old), set(new))
for k in old:
    np.testing.assert_array_equal(new[k], old[k], err_msg=f"load_xrt_spectral_index[{k}]")
print(f"  load_xrt_spectral_index: {len(new['time'])} points identical")

p = {"E_iso": 1.189e52, "n_ism": 18.76, "eps_e": 0.0435, "eps_B": 0.0163, "p": 2.158}
t = np.geomspace(100, 5e5, 200)
o, n = bu.compute_break_frequencies(p, 1.0, t), si.compute_break_frequencies(p, 1.0, t)
for k in ("nu_m", "nu_c"):
    np.testing.assert_array_equal(n[k], o[k], err_msg=f"compute_break_frequencies[{k}]")
print("  compute_break_frequencies identical")

for regime in ("slow", "fast", "both"):
    assert (si.compute_p_prior_from_spectral_index(new, regime)
            == bu.compute_p_prior_from_spectral_index(old, regime)), regime
print("  compute_p_prior_from_spectral_index identical for all 3 regimes")
print("Task 3 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ModuleNotFoundError: No module named 'grb.spectral_index'`**

- [ ] **Step 3: Create `grb/spectral_index.py`**

Copy verbatim from baseline `modeling/utils.py`: `load_xrt_spectral_index` (lines 101-133), `compute_break_frequencies` (136-174), `compute_p_prior_from_spectral_index` (177-217).

Header:

```python
"""XRT spectral-index tooling (Granot & Sari 2001).

Derives the observed spectral index beta = 1 - Gamma from the XRT photon index,
computes synchrotron break frequencies, and turns the measured index into a
Gaussian prior on the electron index p.
"""
import numpy as np
import pandas as pd

from .const import DATA_DIR
```

In `load_xrt_spectral_index`, replace the inline `import pandas as pd` and the
`data_dir is None` block with:

```python
def load_xrt_spectral_index(data_dir=None):
    """... (keep the baseline docstring verbatim) ..."""
    from pathlib import Path
    if data_dir is None:
        data_dir = Path(DATA_DIR)
    else:
        data_dir = Path(data_dir)
```

Everything from `df = pd.read_csv(` onward is verbatim.

> The baseline resolved `data_dir` as `Path(os.path.dirname(__file__)).parent / "data"` from inside `modeling/`, which equals `<repo>/data`. `grb.const.DATA_DIR` is the same directory, so the file read is unchanged — the check in Step 1 proves it.

- [ ] **Step 4: Run the check — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add grb/spectral_index.py modeling/test/check_task3.py
git commit -m "Add grb/spectral_index.py with Granot & Sari XRT index tooling

Verified bitwise-identical to baseline a40dd204.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `grb/params.py`

**Files:**
- Create: `grb/params.py`
- Create: `modeling/test/check_task4.py`

**Interfaces:**
- Consumes: `grb.const.HOST_AV_LOG10_MEAN`, `grb.const.HOST_AV_LOG10_SIGMA`.
- Produces: class `ParamDefWithPrior(name, lower, upper, scale=Scale.LINEAR, initial=None, gaussian_prior=None)` with properties `name, lower, upper, scale, initial` and methods `has_gaussian_prior() -> bool`, `get_prior_mean_sigma() -> tuple`, `to_param_def() -> ParamDef`; `make_param_defs(include_flare=True, include_wing=True) -> list[ParamDefWithPrior]`; `default_nwalkers(ndim) -> int`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task4.py`:

```python
"""Task 4 verification against baseline."""
import sys, os, subprocess, tempfile, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model as bfm   # baseline

from grb.params import make_param_defs, ParamDefWithPrior, default_nwalkers

for flare in (True, False):
    for wing in (True, False):
        old, new = bfm.make_param_defs(flare, wing), make_param_defs(flare, wing)
        assert len(old) == len(new), (flare, wing, len(old), len(new))
        for o, n in zip(old, new):
            assert o.name == n.name, (o.name, n.name)
            assert o.lower == n.lower, (o.name, o.lower, n.lower)
            assert o.upper == n.upper, (o.name, o.upper, n.upper)
            assert o.scale == n.scale, (o.name,)
            assert o.has_gaussian_prior() == n.has_gaussian_prior(), (o.name,)
            assert o.get_prior_mean_sigma() == n.get_prior_mean_sigma(), (o.name,)
        print(f"  flare={flare} wing={wing}: {len(new)} params identical")

assert default_nwalkers(26) == max(4 * 26, 32) == 104
assert default_nwalkers(2) == 32
print("Task 4 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ModuleNotFoundError: No module named 'grb.params'`**

- [ ] **Step 3: Create `grb/params.py`**

```python
"""Parameter definitions for the afterglow fit.

ParamDefWithPrior wraps VegasAfterglow's ParamDef to add an optional Gaussian
prior on top of the box bounds. make_param_defs builds the parameter set for
the combined core + reverse-shock + flare + wing model.
"""
from VegasAfterglow import ParamDef, Scale

from .const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA
```

Then copy verbatim: `ParamDefWithPrior` from baseline `modeling/utils.py:10-47`, `default_nwalkers` from `modeling/utils.py:59-60`, and `make_param_defs` from baseline `modeling/final_model.py:194-250`.

- [ ] **Step 4: Run the check — expect PASS for all four flare/wing combinations**

- [ ] **Step 5: Commit**

```bash
git add grb/params.py modeling/test/check_task4.py
git commit -m "Add grb/params.py with ParamDefWithPrior and make_param_defs

Verified identical bounds, scales, and Gaussian priors against baseline for
all four include_flare/include_wing combinations.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: De-duplicate `grb/modeling.py`

`grb/modeling.py` already holds `load_all_data`, `make_core_model`, and `make_wing_model` as an abandoned near-copy. Make them canonical and correct.

**Files:**
- Modify: `grb/modeling.py`
- Create: `modeling/test/check_task5.py`

**Interfaces:**
- Consumes: `grb.utils.flux_error`, `grb.utils.seconds_from_trigger`, `grb.io.read_data`, `grb.io.filter_data`, `grb.const.{D_L, REDSHIFT, MODEL_RESOLUTIONS}`.
- Produces: `load_all_optical_data() -> tuple[dict, list[dict]]`; `make_core_model(params) -> VegasAfterglow.Model`; `make_wing_model(params) -> VegasAfterglow.Model`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task5.py`:

```python
"""Task 5 verification against baseline."""
import sys, os, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model as bfm

from grb.modeling import load_all_optical_data

o_xrt, o_opt = bfm.load_all_optical_data()
n_xrt, n_opt = load_all_optical_data()

for k in ("time", "flux", "flux_error"):
    np.testing.assert_array_equal(n_xrt[k], o_xrt[k], err_msg=f"xrt[{k}]")
print(f"  XRT: {len(n_xrt['time'])} points identical")

assert len(n_opt) == len(o_opt) == 25, (len(n_opt), len(o_opt))
for a, b in zip(o_opt, n_opt):
    assert a["name"] == b["name"], (a["name"], b["name"])
    assert a["frequency"] == b["frequency"], a["name"]
    for k in ("time", "flux_mJy", "flux_err"):
        np.testing.assert_array_equal(b[k], a[k], err_msg=f"{a['name']}[{k}]")
print(f"  {len(n_opt)} optical datasets identical "
      f"({sum(len(d['time']) for d in n_opt)} points)")

# models must produce identical flux
params = {"E_iso_core": 1.189e52, "Gamma0_core": 522, "theta_c_core": 0.02,
          "n_ism": 18.76, "p": 2.158, "eps_e": 0.0435, "eps_B": 0.0163,
          "xi": 0.943, "tau": 15.0, "p_r": 3.0, "eps_e_r": 0.0422,
          "eps_B_r": 0.20, "xi_r": 0.849, "E_iso_wing": 1e52,
          "Gamma0_wing": 30, "theta_c_wing": 0.3, "p_wing": 2.3,
          "eps_e_wing": 0.9, "eps_B_wing": 0.005, "xi_wing": 0.8}
from grb.modeling import make_core_model, make_wing_model
from grb.utils import model_array
t = np.geomspace(100, 1e5, 20)
for label, mk_new, mk_old in (("core", make_core_model, bfm.make_core_model),
                              ("wing", make_wing_model, bfm.make_wing_model)):
    a = model_array(mk_old(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
    b = model_array(mk_new(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
    np.testing.assert_array_equal(b, a, err_msg=f"{label} flux")
    print(f"  {label} model flux identical")
print("Task 5 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ImportError: cannot import name 'load_all_optical_data'`**

- [ ] **Step 3: Rewrite the import header of `grb/modeling.py`** (replace lines 1-11)

```python
"""Afterglow model construction and observational data assembly.

Builds the VegasAfterglow core-jet (with optional reverse shock) and wing-jet
models, and loads every XRT and optical dataset the fit uses.
"""
import numpy as np

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet

from .const import D_L, REDSHIFT, MODEL_RESOLUTIONS
from .io import read_data, filter_data
from .utils import flux_error, seconds_from_trigger
```

The old header imported `pandas`, `ObsData`, `ParamDef`, `Scale`, `keV`,
`mJy_to_erg_cm2_s_Hz`, and `unit_conversion`. All are unused once
`add_observation` stays commented out — drop them.

- [ ] **Step 4: Rename `load_all_data` to `load_all_optical_data`**

Change the `def` line only. The body already calls `flux_error` and
`seconds_from_trigger`, which now work.

- [ ] **Step 5: Delete the commented-out `add_observation` block** (old lines 64-105)

It is dead, and `modeling.ipynb` already fails importing it. Removing it makes
that failure honest rather than mysterious.

- [ ] **Step 6: Run the check — expect PASS**

If the XRT arrays differ, the cause is `flux_error` vs `xrt_flux_error` — they
are identical implementations, so a difference means the wrong one was wired up.

- [ ] **Step 7: Commit**

```bash
git add grb/modeling.py modeling/test/check_task5.py
git commit -m "Make grb/modeling.py the canonical model + data-loading module

Renames load_all_data to load_all_optical_data, drops unused imports and the
dead commented-out add_observation block. Verified bitwise-identical XRT and
all 25 optical datasets, and identical core/wing model flux, against baseline.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `grb/likelihood.py`

**Files:**
- Create: `grb/likelihood.py`
- Create: `modeling/test/check_task6.py`

**Interfaces:**
- Consumes: `grb.modeling.{make_core_model, make_wing_model}`, `grb.functions.norris_flare`, `grb.extinction.host_extinction_attenuation`, `grb.utils.model_array`, `grb.params.ParamDefWithPrior`, `grb.const.{XRT_BAND, XRT_NU_LO, XRT_NU_HI, SI_FLARE_FRAC_MAX, REDSHIFT}`.
- Produces: `spectral_index_model(core_model, wing_model, params, times, include_flare, flare_frac_max=SI_FLARE_FRAC_MAX) -> tuple[np.ndarray, np.ndarray]`; `spectral_index_chi2(...) -> float`; `compute_model_flux_all_bands(params, xrt_data, optical_datasets, include_flare, include_wing, xrt_index_data=None) -> tuple[np.ndarray, list[np.ndarray], float]`; `log_likelihood(theta, param_defs, xrt_data, optical_datasets, include_flare, include_wing, xrt_index_data=None) -> float`; `log_prior(theta, param_defs) -> float`; `log_probability(...) -> float`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task6.py`:

```python
"""Task 6 verification: likelihood identical to baseline over many theta."""
import sys, os, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model as bfm

from VegasAfterglow import Scale
from grb import likelihood as L
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.spectral_index import load_xrt_spectral_index

pds = make_param_defs(True, True)
xrt, opt = load_all_optical_data()
idx = load_xrt_spectral_index()

def bounds(p):
    lo = np.log10(p.lower) if p.scale is Scale.LOG else p.lower
    hi = np.log10(p.upper) if p.scale is Scale.LOG else p.upper
    return lo, hi

# theta #0: the clipped initial guess main() actually uses
ig = {"E_iso_core":1.189e52,"Gamma0_core":522,"theta_c_core":0.02,"n_ism":18.76,
 "p":2.158,"eps_e":0.0435,"eps_B":0.0163,"xi":0.943,"tau":15.0,"p_r":3.329,
 "eps_e_r":0.0422,"eps_B_r":0.20,"xi_r":0.849,"A_V":0.0254,"t_start_flare":3000,
 "tau_rise_flare":300,"tau_decay_flare":2000,"A_flare":3e-10,"flare_beta":0.8,
 "E_iso_wing":3e51,"Gamma0_wing":30,"theta_c_wing":0.3,"p_wing":2.3,
 "eps_e_wing":0.9,"eps_B_wing":0.005,"xi_wing":0.8}
t0 = []
for p in pds:
    lo, hi = bounds(p)
    v = np.log10(ig[p.name]) if p.scale is Scale.LOG else ig[p.name]
    t0.append(np.clip(v, lo, hi))
thetas = [np.array(t0)]

rng = np.random.default_rng(20260727)
for _ in range(50):
    thetas.append(np.array([rng.uniform(*bounds(p)) for p in pds]))

for i, th in enumerate(thetas):
    o_pri = bfm.log_prior(th, pds); n_pri = L.log_prior(th, pds)
    assert o_pri == n_pri, (i, o_pri, n_pri)
    o_lp = bfm.log_probability(th, pds, xrt, opt, True, True, idx)
    n_lp = L.log_probability(th, pds, xrt, opt, True, True, idx)
    assert o_lp == n_lp, (i, o_lp, n_lp)
    if i % 10 == 0:
        print(f"  theta[{i}]: log_prob={n_lp!r} identical")

# component-level equality at theta[0]
params = {p.name: (10 ** v if p.scale is Scale.LOG else v)
          for p, v in zip(pds, thetas[0])}
o_x, o_o, o_s = bfm.compute_model_flux_all_bands(params, xrt, opt, True, True, idx)
n_x, n_o, n_s = L.compute_model_flux_all_bands(params, xrt, opt, True, True, idx)
np.testing.assert_array_equal(n_x, o_x, err_msg="xrt model flux")
assert n_s == o_s, (n_s, o_s)
assert len(n_o) == len(o_o) == 25
for j, (a, b) in enumerate(zip(o_o, n_o)):
    np.testing.assert_array_equal(b, a, err_msg=f"optical[{j}] {opt[j]['name']}")
print(f"  components identical; si_chi2={n_s!r}")
print(f"Task 6 checks PASSED ({len(thetas)} theta values)")
```

- [ ] **Step 2: Run it, expect `ModuleNotFoundError: No module named 'grb.likelihood'`**

- [ ] **Step 3: Create `grb/likelihood.py`**

Header:

```python
"""Likelihood, prior, and posterior for the combined afterglow fit.

Total flux = core jet (forward + optional reverse shock) + optional wing jet
+ optional Norris flare in the XRT band, spectrally extrapolated to optical.
The likelihood is a chi-squared over XRT plus each optical dataset separately,
plus an optional XRT spectral-index term.
"""
import numpy as np

from VegasAfterglow import Scale

from .const import (
    REDSHIFT,
    SI_FLARE_FRAC_MAX,
    XRT_BAND,
    XRT_NU_HI,
    XRT_NU_LO,
)
from .extinction import host_extinction_attenuation
from .functions import norris_flare
from .modeling import make_core_model, make_wing_model
from .params import ParamDefWithPrior
from .utils import model_array
```

Then copy verbatim from baseline `modeling/final_model.py`: `spectral_index_model` (253-291), `spectral_index_chi2` (294-308), `compute_model_flux_all_bands` (311-392), `log_likelihood` (395-433), `log_prior` (436-451), `log_probability` (454-462).

Two edits inside `compute_model_flux_all_bands`, both pure de-duplication of
values that are already equal:

- Replace the inline `nu_xrt_min = 7.25e16` / `nu_xrt_max = 2.42e18` (baseline
  lines 361-362) with `nu_xrt_min = XRT_NU_LO` and `nu_xrt_max = XRT_NU_HI`.
- Leave every other numeric literal alone.

> `log_likelihood`'s bare `except:` is preserved verbatim. It is load-bearing —
> it converts VegasAfterglow failures into `-inf` so emcee keeps walking.
> Narrowing it would change which parameter vectors survive.

- [ ] **Step 4: Run the check — expect PASS across 51 theta values (~2 min)**

- [ ] **Step 5: Commit**

```bash
git add grb/likelihood.py modeling/test/check_task6.py
git commit -m "Add grb/likelihood.py

log_prior, log_likelihood, log_probability and compute_model_flux_all_bands
verified bitwise-identical to baseline across 51 parameter vectors. XRT band
edges now reference grb.const instead of inline literals.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `grb/results.py`

**Files:**
- Create: `grb/results.py`
- Create: `modeling/test/check_task7.py`

**Interfaces:**
- Consumes: `grb.const.FIT_RESULTS_DIR`.
- Produces: `top_k_samples(samples, log_probs, top_k) -> tuple[np.ndarray, np.ndarray]`; `read_labels(path) -> list[str]`; `latest_result_dir(base_dir, prefix) -> Path`; `load_best_fit_params(outdir) -> tuple[dict, bool, bool]`; `save_run_arrays(outdir, samples, log_probs, labels, top_k=10) -> tuple[np.ndarray, np.ndarray]`; `save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs, xrt_data, optical_datasets) -> None`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task7.py`:

```python
"""Task 7 verification."""
import sys, os, subprocess, tempfile, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from pathlib import Path
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import utils as bu
import final_model_plotting as bfmp

from grb import results as R

rng = np.random.default_rng(7)
s = rng.normal(size=(500, 4)); lp = rng.normal(size=500)
s[10] = s[3]; lp[10] = lp[3]           # force a duplicate to exercise dedup
o_s, o_l = bu.top_k_samples(s, lp, 10); n_s, n_l = R.top_k_samples(s, lp, 10)
np.testing.assert_array_equal(n_s, o_s); np.testing.assert_array_equal(n_l, o_l)
print("  top_k_samples identical (with duplicate present)")

f = Path(tmp) / "labels.txt"; f.write_text("a\n\nb\n c \n")
assert R.read_labels(f) == bu.read_labels(f) == ["a", "b", "c"]
print("  read_labels identical")

from grb.const import FIT_RESULTS_DIR
if FIT_RESULTS_DIR.exists():
    o_d = bu.latest_result_dir(FIT_RESULTS_DIR, "final_")
    n_d = R.latest_result_dir(FIT_RESULTS_DIR, "final_")
    assert o_d == n_d, (o_d, n_d)
    print(f"  latest_result_dir identical -> {n_d.name}")
    o_p, o_f, o_w = bfmp.load_best_fit_params(n_d)
    n_p, n_f, n_w = R.load_best_fit_params(n_d)
    assert (o_f, o_w) == (n_f, n_w)
    assert set(o_p) == set(n_p)
    for k in o_p:
        assert o_p[k] == n_p[k], (k, o_p[k], n_p[k])
    print(f"  load_best_fit_params identical ({len(n_p)} params, "
          f"flare={n_f} wing={n_w})")
else:
    print("  SKIP latest_result_dir/load_best_fit_params: no fit_results/")
print("Task 7 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ModuleNotFoundError: No module named 'grb.results'`**

- [ ] **Step 3: Create `grb/results.py`**

Header:

```python
"""Reading and writing MCMC run artifacts.

Each run writes a directory containing samples.npy, log_probs.npy, labels.txt,
top_k_params.npy, top_k_log_probs.npy and bestfit_params.txt.
"""
from pathlib import Path

import numpy as np

from VegasAfterglow import Scale
```

Copy verbatim from baseline `modeling/utils.py`: `top_k_samples` (63-77),
`read_labels` (97-98), `latest_result_dir` (220-227). Copy
`load_best_fit_params` verbatim from baseline `modeling/final_model_plotting.py`
(34-61).

Then add the two save helpers, lifted from the inline body of baseline
`final_model.py:605-640` with no logic change:

```python
def save_run_arrays(outdir, samples, log_probs, labels, top_k=10):
    """Write samples/log_probs/labels and the top-k subset. Returns (top_params, top_log_probs)."""
    outdir = Path(outdir)
    np.save(outdir / "samples.npy", samples)
    np.save(outdir / "log_probs.npy", log_probs)
    (outdir / "labels.txt").write_text("\n".join(labels))

    top_params, top_log_probs = top_k_samples(samples, log_probs, top_k)
    np.save(outdir / "top_k_params.npy", top_params)
    np.save(outdir / "top_k_log_probs.npy", top_log_probs)
    return top_params, top_log_probs


def save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets):
    """Write the human-readable bestfit_params.txt summary."""
    outdir = Path(outdir)
    lines = [
        "=== Fit Configuration ===",
        f"Model: Core+RS + Norris flare + Wing jet",
        f"XRT data: {len(xrt_data['time'])} points",
    ]
    for dataset in optical_datasets:
        lines.append(f"{dataset['name']}: {len(dataset['time'])} points")
    lines.append(f"Best log probability: {top_log_probs[0]:.6g}")
    lines.append("")
    lines.append("=== Best-fit Parameters ===")
    lines.append(f"{'label':<20} {'sampled':>14} {'physical':>14}")
    lines.append("-" * 50)

    for label, param_def, sampled in zip(labels, param_defs, top_params[0]):
        if param_def.scale is Scale.LOG:
            physical = 10 ** sampled
        else:
            physical = sampled
        lines.append(f"{label:<20} {sampled:>14.6g} {physical:>14.6g}")

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run the check — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add grb/results.py modeling/test/check_task7.py
git commit -m "Add grb/results.py for MCMC run artifacts

Extracts the result-saving logic previously inlined in final_model.main().

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `grb/plotting.py`

**Files:**
- Create: `grb/plotting.py`
- Create: `modeling/test/check_task8.py`

**Interfaces:**
- Consumes: `grb.likelihood.spectral_index_model`, `grb.modeling.{make_core_model, make_wing_model, load_all_optical_data}`, `grb.functions.norris_flare`, `grb.extinction.host_extinction_attenuation`, `grb.results.{load_best_fit_params, read_labels}`, `grb.spectral_index.{load_xrt_spectral_index, compute_break_frequencies}`, `grb.utils.model_array`, `grb.const.{XRT_BAND, REDSHIFT, FIT_RESULTS_DIR}`.
- Produces: `plot_corner(outdir, labels=None, max_samples=20000, seed=42) -> None`; `set_log_y_limits(ax, *values, lower_factor=0.5, upper_factor=2.0) -> None`; `compute_model_components(params, times, frequency, xrt_band, include_flare, include_wing) -> dict` with keys `core_fs, core_rs, flare, wing, total`; `plot_light_curves(outdir) -> Path`; `plot_spectral_index_comparison(outdir) -> Path | None`.

- [ ] **Step 1: Write the failing check**

`modeling/test/check_task8.py`:

```python
"""Task 8 verification: plotted arrays identical to baseline."""
import sys, os, subprocess, tempfile
import matplotlib; matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model_plotting as bfmp

from grb.plotting import compute_model_components, set_log_y_limits
from grb.const import XRT_BAND, FIT_RESULTS_DIR
from grb.results import latest_result_dir, load_best_fit_params

d = latest_result_dir(FIT_RESULTS_DIR, "final_")
params, flare, wing = load_best_fit_params(d)
print(f"  using {d.name} (flare={flare} wing={wing})")

t = np.geomspace(300, 4e5, 60)
# XRT branch
o = bfmp.compute_model_components(params, t, None, XRT_BAND, flare, wing)
n = compute_model_components(params, t, None, XRT_BAND, flare, wing)
for k in ("core_fs", "core_rs", "flare", "wing", "total"):
    np.testing.assert_array_equal(n[k], o[k], err_msg=f"xrt {k}")
print("  XRT components identical")

# optical branch
o = bfmp.compute_model_components(params, t, 3.93e14, None, flare, wing)
n = compute_model_components(params, t, 3.93e14, None, flare, wing)
for k in ("core_fs", "core_rs", "flare", "wing", "total"):
    np.testing.assert_array_equal(n[k], o[k], err_msg=f"optical {k}")
print("  optical components identical")

# set_log_y_limits behaves identically, including the all-non-positive no-op
import matplotlib.pyplot as plt
for vals in ([np.array([1.0, 10.0])], [np.array([-1.0, 0.0, np.nan])]):
    fa = plt.figure().gca(); fb = plt.figure().gca()
    fa.set_yscale("log"); fb.set_yscale("log")
    import utils as bu
    bu.set_log_y_limits(fa, *vals); set_log_y_limits(fb, *vals)
    assert fa.get_ylim() == fb.get_ylim(), (fa.get_ylim(), fb.get_ylim())
    plt.close("all")
print("  set_log_y_limits identical")
print("Task 8 checks PASSED")
```

- [ ] **Step 2: Run it, expect `ModuleNotFoundError: No module named 'grb.plotting'`**

- [ ] **Step 3: Create `grb/plotting.py`**

Header:

```python
"""Figure generation for afterglow fit results.

Light curves with component breakdown, the 7DT spectrum panel, the XRT
spectral-index comparison, and the posterior corner plot.
"""
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from .const import FIT_RESULTS_DIR, REDSHIFT, XRT_BAND, XRT_NU_HI, XRT_NU_LO
from .extinction import host_extinction_attenuation
from .functions import norris_flare
from .likelihood import spectral_index_model
from .modeling import load_all_optical_data, make_core_model, make_wing_model
from .results import load_best_fit_params, read_labels
from .spectral_index import compute_break_frequencies, load_xrt_spectral_index
from .utils import model_array

C_AA_PER_S = 2.99792458e18  # Speed of light in Angstrom/s
```

Copy verbatim: `plot_corner` from baseline `modeling/utils.py:230-262`,
`set_log_y_limits` from `modeling/utils.py:265-276`,
`compute_model_components` from `modeling/final_model_plotting.py:64-155`,
`plot_light_curves` from `:158-353`, `plot_spectral_index_comparison` from
`:356-466`.

Delete these now-unnecessary function-local imports, because the symbols are
imported at module top:

- in `compute_model_components`: `from final_model import (make_core_model, make_wing_model, norris_flare, XRT_BAND)` (baseline lines 69-74)
- in `plot_light_curves`: `from final_model import load_all_optical_data, XRT_BAND` (baseline line 166)
- in `plot_spectral_index_comparison`: `from final_model import make_core_model, make_wing_model, spectral_index_model` (baseline line 392)

Replace the two inline literals in `compute_model_components` (baseline lines
118-119) with `nu_xrt_min = XRT_NU_LO` and `nu_xrt_max = XRT_NU_HI`.

Do **not** change `main()` — it does not move; the CLI lives in Task 9.

- [ ] **Step 4: Run the check — expect PASS**

- [ ] **Step 5: Regenerate a figure and compare bytes against baseline**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
D=$(ls -d modeling/fit_results/final_* | tail -1)
cp "$D/bestfit_lc.png" /tmp/baseline_lc.png
MPLBACKEND=Agg /home/dtak/miniconda3/envs/grb251013c/bin/python -c "
from grb.plotting import plot_light_curves
plot_light_curves('$D')"
cmp /tmp/baseline_lc.png "$D/bestfit_lc.png" && echo "PNG bytes identical" \
  || echo "PNG differs -- arrays already proven identical in Step 4; \
record as a known matplotlib nondeterminism and move on"
```

- [ ] **Step 6: Commit**

```bash
git add grb/plotting.py modeling/test/check_task8.py
git commit -m "Add grb/plotting.py

Breaks the circular final_model <-> final_model_plotting dependency: all
imports are now module-top, no lazy in-function imports. Component arrays
verified bitwise-identical to baseline.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The `fit_final_model.py` driver

**Files:**
- Create: `fit_final_model.py` (repository root)

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: a CLI with flags `--include-flare/--no-include-flare`, `--include-wing/--no-include-wing`, `--use-spectral-index/--no-use-spectral-index`, `--nsteps`, `--nwalkers`, `--ncpus`, `--outdir`.

- [ ] **Step 1: Create `fit_final_model.py`**

```python
#!/usr/bin/env python3
"""Final model: all data, core jet + reverse shock + Norris flare + wing jet.

Thin driver over the grb package. Replaces the former standalone
modeling/final_model.py.

    python fit_final_model.py --ncpus 64
"""
import argparse
import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# Keep each worker single-threaded so pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import emcee
import numpy as np
from VegasAfterglow import Scale

from grb.const import FIT_RESULTS_DIR, SI_FLARE_FRAC_MAX
from grb.likelihood import log_probability
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.plotting import plot_corner, plot_light_curves, plot_spectral_index_comparison
from grb.results import save_bestfit_params, save_run_arrays
from grb.spectral_index import load_xrt_spectral_index

# Initial positions from previous best fits.
# NOTE: p_r (3.329) and E_iso_wing (3e51) fall outside their declared bounds in
# make_param_defs; the clip below pins those walkers to the boundary. Preserved
# verbatim from the original script -- see the refactor design doc.
INITIAL_GUESS = {
    "E_iso_core": 1.189e52, "Gamma0_core": 522, "theta_c_core": 0.02,
    "n_ism": 18.76, "p": 2.158, "eps_e": 0.0435, "eps_B": 0.0163,
    "xi": 0.943, "tau": 15.0, "p_r": 3.329, "eps_e_r": 0.0422,
    "eps_B_r": 0.20, "xi_r": 0.849, "A_V": 0.0254,
    "t_start_flare": 3000, "tau_rise_flare": 300, "tau_decay_flare": 2000,
    "A_flare": 3e-10, "flare_beta": 0.8,
    "E_iso_wing": 3e51, "Gamma0_wing": 30, "theta_c_wing": 0.3,
    "p_wing": 2.3, "eps_e_wing": 0.9, "eps_B_wing": 0.005, "xi_wing": 0.8,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final model: ALL data + Core + Wing + RS + Norris flare")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-wing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-spectral-index", default=True, action=argparse.BooleanOptionalAction,
                        help="Constrain the fit with the XRT spectral index (default: on)")
    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--ncpus", type=int, default=64)
    parser.add_argument("--outdir", default=None)
    return parser.parse_args()


def initial_positions(param_defs, nwalkers):
    """Gaussian around INITIAL_GUESS where known, uniform otherwise, then clipped."""
    pos0 = []
    for p in param_defs:
        if p.name in INITIAL_GUESS:
            center = INITIAL_GUESS[p.name]
            if p.scale is Scale.LOG:
                pos0.append(np.random.normal(np.log10(center), 0.3, nwalkers))
            else:
                pos0.append(np.random.normal(center, center * 0.2, nwalkers))
        else:
            if p.scale is Scale.LOG:
                pos0.append(np.random.uniform(np.log10(p.lower), np.log10(p.upper), nwalkers))
            else:
                pos0.append(np.random.uniform(p.lower, p.upper, nwalkers))

    pos0 = np.array(pos0).T
    for i, p in enumerate(param_defs):
        if p.scale is Scale.LOG:
            pos0[:, i] = np.clip(pos0[:, i], np.log10(p.lower), np.log10(p.upper))
        else:
            pos0[:, i] = np.clip(pos0[:, i], p.lower, p.upper)
    return pos0


def main():
    args = parse_args()

    print("Loading ALL data (XRT + all optical bands)...")
    xrt_data, optical_datasets = load_all_optical_data()

    xrt_index_data = None
    if args.use_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
        except Exception as e:
            print(f"  Warning: could not load XRT spectral index ({e}); continuing without it")

    print(f"\nData loaded:")
    print(f"  XRT: {len(xrt_data['time'])} points "
          f"({xrt_data['time'].min()/3600:.2f}-{xrt_data['time'].max()/3600:.1f} hr)")
    for dataset in optical_datasets:
        print(f"  {dataset['name']}: {len(dataset['time'])} points "
              f"({dataset['time'].min()/3600:.2f}-{dataset['time'].max()/3600:.1f} hr)")

    total_optical = sum(len(d['time']) for d in optical_datasets)
    print(f"\nTotal: {len(xrt_data['time'])} XRT + {total_optical} optical = "
          f"{len(xrt_data['time']) + total_optical} points")
    print(f"Include flare: {args.include_flare}")
    print(f"Include wing: {args.include_wing}")
    if xrt_index_data is not None:
        print(f"XRT spectral index: {len(xrt_index_data['time'])} points "
              f"(applied where core+wing dominate XRT, flare < {SI_FLARE_FRAC_MAX:.0%})")
    else:
        print("XRT spectral index: not used")

    param_defs = make_param_defs(include_flare=args.include_flare,
                                 include_wing=args.include_wing)
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    ndim = len(labels)

    # emcee evaluates walkers in two half-batches, so effective parallelism is
    # capped at nwalkers/2 -> use ~2 walkers per worker.
    n_cpus = mp.cpu_count()
    n_workers = min(args.ncpus, n_cpus - 2)
    nwalkers = args.nwalkers or max(2 * ndim, 2 * n_workers)
    nwalkers += nwalkers % 2
    n_workers = min(n_workers, nwalkers // 2)

    print(f"\nParameters: {ndim}")
    print(f"Walkers: {nwalkers}")
    print(f"Steps: {args.nsteps}")

    pos0 = initial_positions(param_defs, nwalkers)

    print("\nRunning MCMC...")
    print(f"Using {n_workers} CPU cores (out of {n_cpus} available)")
    with Pool(n_workers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(param_defs, xrt_data, optical_datasets,
                  args.include_flare, args.include_wing, xrt_index_data),
            pool=pool,
        )
        sampler.run_mcmc(pos0, args.nsteps, progress=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = "final"
    if args.include_flare:
        phase_name += "_flare"
    if args.include_wing:
        phase_name += "_wing"
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"{phase_name}_{run_ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    top_params, top_log_probs = save_run_arrays(outdir, samples, log_probs, labels)

    print(f"\nBest log probability: {top_log_probs[0]:.3f}")
    print(f"Results saved to: {outdir}")

    save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets)

    print("\nPlotting best-fit light curves...")
    plot_light_curves(outdir)

    if xrt_index_data is not None:
        print("Plotting spectral index comparison...")
        plot_spectral_index_comparison(outdir)

    plot_corner(outdir, labels, max_samples=20000)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")

    print("\n" + "=" * 60)
    print("✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the driver's setup phase matches baseline**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
/home/dtak/miniconda3/envs/grb251013c/bin/python -c "
import fit_final_model as d
from VegasAfterglow import Scale
pds = d.make_param_defs(True, True)
labels = [f'log10_{p.name}' if p.scale is Scale.LOG else p.name for p in pds]
assert len(labels) == 26, len(labels)
print('ndim', len(labels)); print(labels)
p0 = d.initial_positions(pds, 104)
assert p0.shape == (104, 26), p0.shape
import numpy as np
for i, p in enumerate(pds):
    lo = np.log10(p.lower) if p.scale is Scale.LOG else p.lower
    hi = np.log10(p.upper) if p.scale is Scale.LOG else p.upper
    assert (p0[:, i] >= lo).all() and (p0[:, i] <= hi).all(), p.name
print('initial_positions in-bounds OK')
"
```

Expected: `ndim 26`, the 26 labels, `initial_positions in-bounds OK`.

- [ ] **Step 3: Run a 2-step end-to-end smoke test**

```bash
/home/dtak/miniconda3/envs/grb251013c/bin/python fit_final_model.py \
  --nsteps 2 --nwalkers 4 --ncpus 2 --outdir /tmp/grb_smoke
ls -1 /tmp/grb_smoke
```

Expected files: `samples.npy`, `log_probs.npy`, `labels.txt`, `top_k_params.npy`,
`top_k_log_probs.npy`, `bestfit_params.txt`, `bestfit_lc.png`,
`spectral_index_comparison.png`, `corner_plot.png`.

> `corner` may warn about too few samples at nwalkers=4; a warning is fine, a
> traceback is not.

- [ ] **Step 4: Commit**

```bash
git add fit_final_model.py
git commit -m "Add fit_final_model.py driver built on the grb package

Replaces the 659-line standalone modeling/final_model.py with a thin CLI.
Same flags, same output layout, same fit_results/<phase>_<ts>/ destination.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Consolidated equivalence harness

**Files:**
- Create: `modeling/test/test_refactor_equivalence.py`
- Delete: `modeling/test/check_task{1..8}.py`

**Interfaces:**
- Consumes: the whole `grb` package.
- Produces: an exit-code-0-on-success script.

- [ ] **Step 1: Write `modeling/test/test_refactor_equivalence.py`**

Merge the per-task checks into one script covering the six spec assertions:
data loading, optical datasets, spectral index, param defs, the 51-theta
`log_prior`/`log_likelihood`/`log_probability` sweep, and
`compute_model_flux_all_bands` components. Reuse the git-materialisation
preamble from Task 6 Step 1 verbatim. Print a per-section PASS line and exit 1
on any failure.

The script must set `matplotlib.use("Agg")` before importing `grb.plotting`,
and must not depend on `modeling/fit_results/` existing — skip the
`latest_result_dir` section with a printed SKIP if it is absent.

- [ ] **Step 2: Run it — expect every section PASS**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py
echo "exit=$?"
```

Expected: `exit=0` and a final `ALL EQUIVALENCE CHECKS PASSED`.

- [ ] **Step 3: Delete the scaffolding checks**

```bash
git rm modeling/test/check_task1.py modeling/test/check_task2.py \
       modeling/test/check_task3.py modeling/test/check_task4.py \
       modeling/test/check_task5.py modeling/test/check_task6.py \
       modeling/test/check_task7.py modeling/test/check_task8.py
```

- [ ] **Step 4: Commit**

```bash
git add modeling/test/test_refactor_equivalence.py
git commit -m "Add consolidated refactor equivalence harness

Proves the grb package reproduces baseline a40dd204 bitwise: identical data
arrays, param defs, and log_probability across 51 parameter vectors.
Supersedes the per-task scaffolding checks.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Migrate the nine callers and delete the old modules

**Files:**
- Modify: `modeling/early_phase.py:23-33`, `modeling/early_phase_plotting.py:13-22`, `modeling/late_phase.py:21-32`, `modeling/late_phase_plotting.py:13-22`, `modeling/partial_data.py:18-29`, `modeling/partial_data_plotting.py:18-23`, `modeling/test/test_jet_break_spreading.py`, `modeling/test/test_spreading.py`, `modeling/test/test_narrow_jet_final.py`
- Delete: `modeling/utils.py`, `modeling/final_model.py`, `modeling/final_model_plotting.py`

**Interfaces:**
- Consumes: all `grb.*` modules from Tasks 1-8.
- Produces: nothing new.

- [ ] **Step 1: Apply this import rewrite to all nine files**

Old name on the left, new import on the right:

| Old `from utils import X` | New |
|---|---|
| `HOST_AV_LOG10_MEAN`, `HOST_AV_LOG10_SIGMA`, `XRT_EXCLUDE_TIME_RANGE`, `XRT_FLARE_START_TIME` | `from grb.const import ...` |
| `ParamDefWithPrior`, `default_nwalkers` | `from grb.params import ...` |
| `load_xrt_spectral_index`, `compute_break_frequencies`, `compute_p_prior_from_spectral_index` | `from grb.spectral_index import ...` |
| `host_extinction_attenuation` | `from grb.extinction import ...` |
| `model_array` | `from grb.utils import ...` |
| `xrt_flux_error` | `from grb.utils import flux_error` — **and rename every call site** |
| `plot_corner`, `set_log_y_limits` | `from grb.plotting import ...` |
| `top_k_samples`, `read_labels`, `latest_result_dir` | `from grb.results import ...` |

Concretely, `modeling/early_phase.py:23-33` becomes:

```python
from grb.const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA, XRT_FLARE_START_TIME
from grb.params import ParamDefWithPrior
from grb.plotting import plot_corner
from grb.spectral_index import (
    compute_break_frequencies,
    compute_p_prior_from_spectral_index,
    load_xrt_spectral_index,
)
from grb.utils import flux_error
```

and `modeling/late_phase.py:21-32` becomes:

```python
from grb.const import (
    HOST_AV_LOG10_MEAN,
    HOST_AV_LOG10_SIGMA,
    XRT_EXCLUDE_TIME_RANGE,
)
from grb.extinction import host_extinction_attenuation
from grb.results import read_labels, top_k_samples
from grb.spectral_index import compute_break_frequencies, load_xrt_spectral_index
from grb.utils import flux_error
```

Note `default_nwalkers` is **dropped** from `late_phase.py` and
`partial_data.py` — both import it and never call it. Leave their inlined
`max(int(2.5 * ndim), 32)` expressions untouched.

- [ ] **Step 2: Rename `xrt_flux_error` call sites**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
grep -rn 'xrt_flux_error' modeling/ | grep -v test_refactor_equivalence
```

Replace each remaining `xrt_flux_error(` call with `flux_error(`. Re-run the
grep; it must return nothing outside the equivalence harness.

- [ ] **Step 3: Delete the three moved modules**

```bash
git rm modeling/utils.py modeling/final_model.py modeling/final_model_plotting.py
```

- [ ] **Step 4: Verify every migrated file imports cleanly**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
for m in early_phase early_phase_plotting late_phase late_phase_plotting \
         partial_data partial_data_plotting spectral_index_interpolator; do
  MPLBACKEND=Agg PYTHONPATH=".:modeling" \
    /home/dtak/miniconda3/envs/grb251013c/bin/python -c "import $m" \
    && echo "OK   $m" || echo "FAIL $m"
done
for m in test_jet_break test_jet_break_spreading test_spreading \
         test_narrow_jet_final demo_smooth_spectral_index_standalone; do
  MPLBACKEND=Agg PYTHONPATH=".:modeling:modeling/test" \
    /home/dtak/miniconda3/envs/grb251013c/bin/python -c "import $m" \
    && echo "OK   $m" || echo "FAIL $m"
done
```

Expected: `OK` for all twelve. Any `FAIL` must be fixed before committing.

> These modules run argparse and heavy work only under `if __name__ ==
> "__main__"`, so importing them is safe and does not start a fit.

- [ ] **Step 5: Confirm no stale references remain**

```bash
grep -rn '^from utils import\|^import utils\|from final_model import\|from final_model_plotting import' \
  modeling/ *.py 2>/dev/null || echo "no stale references"
```

- [ ] **Step 6: Re-run the equivalence harness**

```bash
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py
echo "exit=$?"
```

Expected: `exit=0`. It reads the baseline from git, so deleting the working-tree
copies does not affect it.

- [ ] **Step 7: Commit**

```bash
git add -A modeling/
git commit -m "Migrate all callers to grb.* and delete the moved modules

Removes modeling/utils.py, modeling/final_model.py and
modeling/final_model_plotting.py. Nine callers now import from grb.const,
grb.params, grb.spectral_index, grb.extinction, grb.utils, grb.plotting and
grb.results. Drops the dead default_nwalkers imports from late_phase.py and
partial_data.py, and renames xrt_flux_error call sites to flux_error.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Final verification and documentation

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/superpowers/reports/2026-07-27-fragmentation-audit.md`

- [ ] **Step 1: Assert every success criterion mechanically**

```bash
cd /data/dtak/research/grb/GRB251013C/.claude/worktrees/grb-module-refactor
/home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py || exit 1

test ! -e modeling/final_model.py && test ! -e modeling/final_model_plotting.py \
  && test ! -e modeling/utils.py && echo "OK old modules gone"

grep -rn 'from modeling\|import modeling\b' grb/ && echo "FAIL grb imports modeling" \
  || echo "OK grb has no modeling import"

for sym in norris_flare make_core_model make_wing_model load_all_optical_data \
           host_extinction_attenuation model_array top_k_samples ParamDefWithPrior; do
  c=$(grep -rn "^def $sym\|^class $sym" grb/ modeling/ *.py 2>/dev/null | wc -l)
  [ "$c" -eq 1 ] && echo "OK  $sym defined once" || echo "FAIL $sym defined $c times"
done
```

Every line must read `OK`.

- [ ] **Step 2: Update `CLAUDE.md`**

In the architecture section, replace the description of `modeling/final_model.py`
as the primary fit with: the primary fit is now `fit_final_model.py` at the
repository root, run from the root rather than from `modeling/`; the physics,
likelihood, parameters, results I/O and plotting live in `grb/`. List the five
new modules. Note that the other phase scripts still run from `modeling/`.
Correct the stale `VegasAfterglow==1.1.0` note to state 2.0.6 is installed.

- [ ] **Step 3: Write the fragmentation audit report**

`docs/superpowers/reports/2026-07-27-fragmentation-audit.md` recording every
finding: the six duplicated symbols with similarity scores, the two bugs fixed,
and the six deferred items from the spec's Deferred section — the MW/SMC
extinction disagreement, the duplicate `xrt_index.csv` readers, the mislabelled
`host_galaxy_extinction_prior`, the two out-of-bounds initial guesses, the dead
`default_nwalkers` imports, and the triplicated XRT band edges. For each
deferred item state what is wrong, why it was not changed, and what deciding it
would require.

- [ ] **Step 4: Commit and push**

```bash
git add CLAUDE.md docs/
git commit -m "Update CLAUDE.md and add the fragmentation audit report

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push -u origin worktree-grb-module-refactor
gh pr create --draft --title "Refactor final model into the grb package" \
  --body "$(cat <<'BODY'
Moves `modeling/final_model.py`, `modeling/final_model_plotting.py` and
`modeling/utils.py` into the `grb` package, replacing them with a thin
`fit_final_model.py` driver.

Results are bitwise identical to baseline `a40dd204`, proven by
`modeling/test/test_refactor_equivalence.py`.

See `docs/superpowers/specs/2026-07-27-grb-module-refactor-design.md` for the
design and `docs/superpowers/reports/2026-07-27-fragmentation-audit.md` for
the findings, including six issues reported but deliberately left unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Self-Review

**Spec coverage.** Architecture → Tasks 1-8. Driver → Task 9. Equivalence
verification (all six assertions) → Task 10. Callers to migrate → Task 11.
Bugs fixed → Task 1 (both). Deferred items → Task 12 Step 3. Success criteria
1-6 → Task 12 Step 1. No gaps.

**Placeholder scan.** No TBD/TODO. Every code step carries real code or an exact
line range to copy from a named baseline file. Verification steps carry runnable
commands with stated expected output.

**Type consistency.** `load_all_optical_data` is the single name throughout
(Tasks 5, 6, 8, 9) — the baseline's `load_all_data` is renamed once in Task 5.
`flux_error` is the single name after Task 11; `xrt_flux_error` survives only in
Task 6's baseline-comparison code, which is correct because it names the
baseline symbol. `save_run_arrays` and `save_bestfit_params` are defined in
Task 7 and consumed in Task 9 with matching signatures.

**Known risk.** Task 8 Step 5 compares PNG bytes, which may differ for reasons
unrelated to correctness. It is deliberately non-blocking: the arrays behind the
figure are already proven identical in Step 4.
