"""Task 7 verification."""
import sys, os, subprocess, tempfile
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
