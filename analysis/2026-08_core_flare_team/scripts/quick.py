import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json, itertools
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C")
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index, HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA
from scipy.optimize import minimize
from multiprocessing import Pool
xrt_data,optical_datasets=fm.load_all_optical_data()
xrt_index_data=load_xrt_spectral_index()
pdefs=fm.make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is fm.Scale.LOG for p in pdefs]
d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy");lp=np.load(d+"/log_probs.npy");best=s[np.argmax(lp)].copy()
iav=IX["log10_A_V"]
def full_target(th):
    r=fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    if not np.isfinite(r): return -1e12
    return r-0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
# free subset: core jet + microphysics (+ A_V), everything else frozen at 171919 best
FREE=["log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism","p",
      "log10_eps_e","log10_eps_B","log10_A_V"]
FB={"log10_E_iso_core":(np.log10(1e51),np.log10(1e54)),
    "log10_Gamma0_core":(np.log10(100),np.log10(2000)),
    "log10_theta_c_core":(np.log10(0.005),np.log10(0.3)),
    "log10_n_ism":(np.log10(0.01),np.log10(600)),
    "p":(2.01,3.0),
    "log10_eps_e":(np.log10(0.005),np.log10(0.5)),
    "log10_eps_B":(np.log10(1e-7),np.log10(0.1)),
    "log10_A_V":(np.log10(0.001),np.log10(2.0))}
idx=[IX[f] for f in FREE]; bb=np.array([FB[f] for f in FREE])
def run(job):
    tag,seed=job
    def f(x):
        th=best.copy(); th[idx]=np.clip(x,bb[:,0],bb[:,1])
        return -full_target(th)
    r=minimize(f,np.clip(seed,bb[:,0],bb[:,1]),method="Powell",
               bounds=list(map(tuple,bb)),options=dict(maxfev=8000,xtol=1e-4,ftol=1e-7))
    th=best.copy(); th[idx]=r.x
    return tag,float(-r.fun),th.tolist()
seeds=[("A_current",best[idx].copy())]
for p_,eb,n_ in itertools.product([2.2,2.4,2.6],[1e-4,1e-5,1e-6],[3.0,30.0,140.0]):
    x=best[idx].copy()
    x[FREE.index("p")]=p_
    x[FREE.index("log10_eps_B")]=np.log10(eb)
    x[FREE.index("log10_n_ism")]=np.log10(n_)
    x[FREE.index("log10_E_iso_core")]=np.log10(3e52)
    x[FREE.index("log10_eps_e")]=np.log10(0.12)
    seeds.append((f"B_p{p_}_eB{eb:g}_n{n_:g}",x))
with Pool(28) as pool: out=pool.map(run,seeds)
out.sort(key=lambda x:-x[1])
print(f"reference (171919 best) target = {full_target(best):.2f}\n")
for t,v,x in out[:12]: print(f"{t:<26} target={v:>10.2f}")
json.dump({"labels":labels,"results":out},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/quick.json","w"))
t,v,x=out[0]; x=np.array(x)
print(f"\n=== BEST {t} target={v:.2f} (vs {full_target(best):.2f}) ===")
for i,l in enumerate(labels):
    ch=" <-- changed" if abs(x[i]-best[i])>1e-6 else ""
    print(f"{l:<24}{x[i]:>10.4f}{(10**x[i] if LOGP[i] else x[i]):>14.5g}{ch}")
