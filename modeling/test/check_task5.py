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

from grb.modeling import load_all_optical_data, make_core_model, make_wing_model
from grb.utils import model_array

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

params = {"E_iso_core": 1.189e52, "Gamma0_core": 522, "theta_c_core": 0.02,
          "n_ism": 18.76, "p": 2.158, "eps_e": 0.0435, "eps_B": 0.0163,
          "xi": 0.943, "tau": 15.0, "p_r": 3.0, "eps_e_r": 0.0422,
          "eps_B_r": 0.20, "xi_r": 0.849, "E_iso_wing": 1e52,
          "Gamma0_wing": 30, "theta_c_wing": 0.3, "p_wing": 2.3,
          "eps_e_wing": 0.9, "eps_B_wing": 0.005, "xi_wing": 0.8}
t = np.geomspace(100, 1e5, 20)
for label, mk_new, mk_old in (("core", make_core_model, bfm.make_core_model),
                              ("wing", make_wing_model, bfm.make_wing_model)):
    a = model_array(mk_old(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
    b = model_array(mk_new(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
    np.testing.assert_array_equal(b, a, err_msg=f"{label} flux")
    print(f"  {label} model flux identical")
print("Task 5 checks PASSED")
