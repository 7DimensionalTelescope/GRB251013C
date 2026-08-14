"""Monotone Powell polish (no scipy bounds; box enforced via penalty) of the probe best."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_likelihood
from grb.spectral_index import load_xrt_spectral_index
from grb.const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA
from scipy.optimize import minimize

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]

B=np.zeros((len(labels),2))
for i,p in enumerate(pdefs):
    B[i]=[np.log10(p.lower),np.log10(p.upper)] if LOGP[i] else [p.lower,p.upper]
# the widened box being shipped
B[IX["p"]]=[1.6,2.3]
B[IX["p_wing"]]=[1.8,3.3]
B[IX["log10_n_ism"]][1]=np.log10(1000)

x0=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/probe_best.npy")
iav=IX["log10_A_V"]
def target(th):
    if np.any(th<B[:,0]) or np.any(th>B[:,1]): return 1e12
    r=log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    if not np.isfinite(r): return 1e12
    r += -0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
    return -r

v0=-target(x0)
print(f"start logP = {v0:.2f}", flush=True)
r=minimize(target,x0,method="Powell",options=dict(maxfev=20000,xtol=1e-4,ftol=1e-7))
xf=np.clip(r.x,B[:,0],B[:,1])
vf=-target(xf)
print(f"polished logP = {vf:.2f}  ({r.nfev} evals)")
best = xf if vf>=v0 else x0
print(f"using {'polished' if vf>=v0 else 'probe'} vector, logP={max(vf,v0):.2f}\n")
for i,l in enumerate(labels):
    v=best[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/polished_best.npy",best)
json.dump({"logp":float(max(vf,v0))},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/polish.json","w"))
