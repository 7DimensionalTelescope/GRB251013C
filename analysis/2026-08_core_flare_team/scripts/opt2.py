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
from multiprocessing import Pool

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]

# Exploration box = current retuned bounds, EXCEPT:
#  - p, p_wing opened below 2 (the experiment)
#  - n_ism, theta_c_wing rails released so they don't confound the p test
B=np.zeros((len(labels),2))
for i,p in enumerate(pdefs):
    B[i]=[np.log10(p.lower),np.log10(p.upper)] if LOGP[i] else [p.lower,p.upper]
B[IX["p"]]        =[1.5,2.3]
B[IX["p_wing"]]   =[1.5,3.3]
B[IX["log10_n_ism"]][1]=np.log10(1000)
B[IX["log10_theta_c_wing"]][1]=np.log10(1.0)

RUN="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260730_171914"
best=np.load(RUN+"/top_k_params.npy")[0].copy()

iav=IX["log10_A_V"]
def target(th):
    if np.any(th<B[:,0]) or np.any(th>B[:,1]): return 1e12
    r=log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    if not np.isfinite(r): return 1e12
    r += -0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
    return -r

def seed(**kw):
    x=best.copy()
    for k,v in kw.items():
        x[IX[k]]=v
    return x

jobs=[
  ("control", best.copy()),
  ("p1.85",            seed(p=1.85)),
  ("pw1.9",            seed(p_wing=1.9)),
  ("p1.85_pw1.9",      seed(p=1.85, p_wing=1.9)),
  ("p1.7_pw1.7",       seed(p=1.7, p_wing=1.7)),
  ("pw2.4",            seed(p_wing=2.4)),
  ("p1.85_pw2.4",      seed(p=1.85, p_wing=2.4)),
  ("p1.85_pw1.9_hiEe", seed(p=1.85, p_wing=1.9, log10_eps_e=np.log10(0.09))),
]

def run(job):
    tag,x0=job
    x0=np.clip(np.asarray(x0,float),B[:,0],B[:,1])
    v0=-target(x0)
    r=minimize(target,x0,method="Powell",bounds=list(map(tuple,B)),
               options=dict(maxfev=40000,xtol=1e-4,ftol=1e-6))
    print(f"[done] {tag}: start {v0:.2f} -> {-r.fun:.2f} ({r.nfev} evals)", flush=True)
    return tag,float(-r.fun),r.x.tolist()

print(f"{len(jobs)} starts; control target at start = {-target(best):.2f}", flush=True)
with Pool(len(jobs)) as pool: out=pool.map(run,jobs)
out.sort(key=lambda x:-x[1])
json.dump({"labels":labels,"B":B.tolist(),"results":out},
          open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/opt2.json","w"))
print()
for tag,v,x in out:
    xx=np.array(x)
    print(f"{tag:<18} logP={v:>9.2f}  p={xx[IX['p']]:.3f}  p_wing={xx[IX['p_wing']]:.3f}  "
          f"n={10**xx[IX['log10_n_ism']]:.0f}  th_w={10**xx[IX['log10_theta_c_wing']]:.2f}")
tag,v,x=out[0]; x=np.array(x)
print(f"\n=== BEST {tag}  logP={v:.2f} ===")
for i,l in enumerate(labels):
    print(f"{l:<24}{x[i]:>10.4f}{(10**x[i] if LOGP[i] else x[i]):>14.5g}")
