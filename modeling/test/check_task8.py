"""Task 8 verification: grb.plotting vs baseline modeling/final_model_plotting.py.

Default run compares compute_model_components arrays (fast, decisive).
Pass --png to additionally render both sides' figures and compare bytes
(slow: ~300 model builds per side; non-blocking, reported only).
"""
import sys, os, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

# Mirror the real layout (<base>/modeling/*.py, <base>/data) because baseline
# utils.load_xrt_spectral_index resolves data/ two levels up from its own __file__.
base = tempfile.mkdtemp()
tmp = os.path.join(base, "modeling")
os.mkdir(tmp)
os.symlink(os.path.join(ROOT, "data"), os.path.join(base, "data"))
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model_plotting as bfmp

from grb import plotting as P
from grb.const import FIT_RESULTS_DIR, XRT_BAND
from grb.results import latest_result_dir, load_best_fit_params

result_dir = latest_result_dir(FIT_RESULTS_DIR, "final_")
params, include_flare, include_wing = load_best_fit_params(result_dir)
print(f"  using {result_dir.name} (flare={include_flare} wing={include_wing})")

t = np.geomspace(1e3, 3e5, 12)

# XRT branch
o = bfmp.compute_model_components(params, t, None, XRT_BAND, include_flare, include_wing)
n = P.compute_model_components(params, t, None, XRT_BAND, include_flare, include_wing)
assert set(o) == set(n)
for k in o:
    np.testing.assert_array_equal(np.asarray(n[k]), np.asarray(o[k]))
print(f"  XRT components identical ({sorted(o)}), total[0]={n['total'][0]:.10g}")

# Optical branch (i-band frequency)
nu = 3.9e14
o = bfmp.compute_model_components(params, t, nu, None, include_flare, include_wing)
n = P.compute_model_components(params, t, nu, None, include_flare, include_wing)
for k in o:
    np.testing.assert_array_equal(np.asarray(n[k]), np.asarray(o[k]))
print(f"  optical components identical, total[0]={n['total'][0]:.10g}")

# set_log_y_limits
import matplotlib.pyplot as plt
vals = [np.array([1e-3, -5.0, np.nan, 2.0]), np.array([np.inf, 7.0])]
figs = []
lims = []
for mod in (bfmp, P):
    f, ax = plt.subplots()
    ax.set_yscale("log")
    mod.set_log_y_limits(ax, *vals)
    lims.append(ax.get_ylim())
    plt.close(f)
assert lims[0] == lims[1], lims
print(f"  set_log_y_limits identical -> {lims[0]}")

if "--png" in sys.argv:
    def stage(tag):
        d = Path(tempfile.mkdtemp(prefix=f"lc_{tag}_"))
        for f in ("samples.npy", "log_probs.npy", "labels.txt"):
            os.symlink(result_dir / f, d / f)
        return d

    d_old, d_new = stage("old"), stage("new")
    bfmp.plot_light_curves(d_old)
    P.plot_light_curves(d_new)
    bfmp.plot_spectral_index_comparison(d_old)
    P.plot_spectral_index_comparison(d_new)
    for name in ("bestfit_lc.png", "spectral_index_comparison.png"):
        a = (d_old / name).read_bytes()
        b = (d_new / name).read_bytes()
        print(f"  PNG {name}: {'byte-identical' if a == b else f'DIFFER ({len(a)} vs {len(b)} bytes)'}")

print("Task 8 checks PASSED")
