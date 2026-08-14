"""Fast no-wing joint test: wide single-power-law core (frozen per branch)
+ RS + flare + A_V vs ALL data. include_wing=False (true 19-param model)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json, time
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
pdefs = make_param_defs(True, False)   # no wing: 19 params
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]
assert len(labels)==19

INC=np.load("/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026/top_k_params.npy")[0][:19]

good=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_good.npy")
glp=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_lp.npy")
CFREE=["log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism","p",
       "log10_eps_e","log10_eps_B","xi",
       "log10_t_start_flare","log10_tau_rise_flare","log10_tau_decay_flare",
       "log10_A_flare","flare_beta"]
CIX={l:i for i,l in enumerate(CFREE)}
eB=good[:,CIX["log10_eps_B"]]; pv=good[:,CIX["p"]]
bSC=eB<-3.5; bHP=(~bSC)&(pv<2.0); bMID=(~bSC)&(~bHP)
def base_theta(mask):
    x=INC.copy()
    cb=good[mask][np.argmax(glp[mask])]
    for l in CFREE:
        x[IX[l]]=cb[CIX[l]]
    return x

FREE=list(labels)   # all 19: core + RS + flare + A_V all vary
FI=[IX[l] for l in FREE]
SPEC={
 "log10_E_iso_core":(1e51,1e55),"log10_Gamma0_core":(100,2000),
 "log10_theta_c_core":(0.02,0.8),"log10_n_ism":(0.01,3000),
 "p":(1.6,3.2),"log10_eps_e":(0.001,0.5),"log10_eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "log10_tau":(5,100),"p_r":(2.0,3.5),"log10_eps_e_r":(0.005,0.5),
 "log10_eps_B_r":(1e-4,0.6),"xi_r":(0.3,1.0),
 "log10_t_start_flare":(1000,5000),"log10_tau_rise_flare":(5,2000),
 "log10_tau_decay_flare":(500,2e4),"log10_A_flare":(1e-11,5e-9),"flare_beta":(0.3,1.5),
 "log10_A_V":(0.001,2.0),
}
BF=np.array([[np.log10(SPEC[l][0]),np.log10(SPEC[l][1])] if l.startswith("log10_")
             else list(SPEC[l]) for l in FREE])
iav=FREE.index("log10_A_V")

def make_target(base):
    def target(free):
        if np.any(free<BF[:,0]) or np.any(free>BF[:,1]): return 1e12
        th=base.copy(); th[FI]=free
        r=log_likelihood(th,pdefs,xrt_data,optical_datasets,True,False,xrt_index_data)
        if not np.isfinite(r): return 1e12
        r += -0.5*((free[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
        return -r
    return target

def run(job):
    tag,mask=job
    base=base_theta(mask)
    target=make_target(base)
    x0=np.clip(base[FI],BF[:,0]+1e-6,BF[:,1]-1e-6)
    v0=-target(x0); t0=time.time()
    r=minimize(target,x0,method="Powell",options=dict(maxfev=9000,xtol=1e-3,ftol=1e-6))
    th=base.copy(); th[FI]=np.clip(r.x,BF[:,0],BF[:,1])
    print(f"[done] {tag}: {v0:.1f} -> {-r.fun:.1f} ({r.nfev} evals, {time.time()-t0:.0f}s)",flush=True)
    return tag,float(-r.fun),th.tolist()

print("no-wing test: wide core frozen per branch; free RS+flare+A_V",flush=True)
with Pool(3) as pool: out=pool.map(run,[("HP",bHP),("SC",bSC),("MID",bMID)])
out.sort(key=lambda x:-x[1])
json.dump({"labels":labels,"results":out},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/jointfast_nowing.json","w"))
for tag,v,_ in out: print(f"{tag}: logP = {v:.2f}")
