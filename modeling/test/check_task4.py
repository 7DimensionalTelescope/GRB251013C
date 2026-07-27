"""Task 4 verification against baseline."""
import sys, os, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

tmp = tempfile.mkdtemp()
for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
    open(os.path.join(tmp, name), "wb").write(blob)
sys.path.insert(0, tmp)
import final_model as bfm

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
