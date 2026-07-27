"""Task 3 verification. Compares against the baseline implementation."""
import sys, os, subprocess, tempfile, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/utils.py"])
open(os.path.join(tmp, "baseline_utils.py"), "wb").write(src)
spec = importlib.util.spec_from_file_location("baseline_utils",
                                              os.path.join(tmp, "baseline_utils.py"))
bu = importlib.util.module_from_spec(spec); spec.loader.exec_module(bu)

from grb import spectral_index as si

from pathlib import Path
DATA = Path(ROOT) / "data"
old = bu.load_xrt_spectral_index(data_dir=DATA); new = si.load_xrt_spectral_index()
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
