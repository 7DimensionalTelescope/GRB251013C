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

pds = make_param_defs(True, True)          # grb implementation
bpds = bfm.make_param_defs(True, True)     # baseline implementation
# Each side must use its OWN ParamDefWithPrior class: baseline log_prior does an
# isinstance() check against the class it imported, so feeding it grb-built defs
# would silently skip the Gaussian prior. Task 4 proved the two sets are identical.
xrt, opt = load_all_optical_data()
idx = load_xrt_spectral_index()

def bounds(p):
    lo = np.log10(p.lower) if p.scale is Scale.LOG else p.lower
    hi = np.log10(p.upper) if p.scale is Scale.LOG else p.upper
    return lo, hi

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

n_finite = 0
for i, th in enumerate(thetas):
    o_pri = bfm.log_prior(th, bpds); n_pri = L.log_prior(th, pds)
    assert o_pri == n_pri, (i, o_pri, n_pri)
    o_lp = bfm.log_probability(th, bpds, xrt, opt, True, True, idx)
    n_lp = L.log_probability(th, pds, xrt, opt, True, True, idx)
    assert o_lp == n_lp, (i, o_lp, n_lp)
    if np.isfinite(n_lp): n_finite += 1
    if i % 10 == 0:
        print(f"  theta[{i}]: log_prob={n_lp!r} identical")

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
print(f"Task 6 checks PASSED ({len(thetas)} theta, {n_finite} finite log_prob)")
