import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json, itertools
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index, HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA
from scipy.optimize import minimize
from multiprocessing import Pool

xrt_data, optical_datasets = fm.load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = fm.make_param_defs(True, True)
labels = [f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
IX = {l:i for i,l in enumerate(labels)}
LOGP = [p.scale is fm.Scale.LOG for p in pdefs]

# broad exploration box (sampled space)
SPEC = {  # name: (lo, hi) in PHYSICAL units for LOG params, sampled for linear
 "log10_E_iso_core":(1e51,1e54), "log10_Gamma0_core":(100,2000),
 "log10_theta_c_core":(0.005,0.3), "log10_n_ism":(0.01,600),
 "p":(2.01,3.0), "log10_eps_e":(0.005,0.5), "log10_eps_B":(1e-6,0.1),
 "xi":(0.3,1.0), "log10_tau":(2,100),
 "p_r":(2.0,3.6), "log10_eps_e_r":(0.005,0.5), "log10_eps_B_r":(1e-4,0.6), "xi_r":(0.3,1.0),
 "log10_A_V":(0.001,2.0),
 "log10_t_start_flare":(1000,6000), "log10_tau_rise_flare":(20,3000),
 "log10_tau_decay_flare":(500,20000), "log10_A_flare":(1e-11,1e-8), "flare_beta":(0.3,1.5),
 "log10_E_iso_wing":(1e50,2e53), "log10_Gamma0_wing":(5,150),
 "log10_theta_c_wing":(0.05,1.0), "p_wing":(2.1,3.4),
 "log10_eps_e_wing":(0.005,1.0), "log10_eps_B_wing":(1e-5,0.1), "xi_wing":(0.3,1.0),
}
B=np.zeros((len(labels),2))
for l,(lo,hi) in SPEC.items():
    i=IX[l]; B[i]=[np.log10(lo),np.log10(hi)] if LOGP[i] else [lo,hi]

d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy"); lp=np.load(d+"/log_probs.npy")
best=s[np.argmax(lp)].copy()

iav=IX["log10_A_V"]
def target(th):
    if np.any(th<B[:,0]) or np.any(th>B[:,1]): return 1e12
    r = fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    if not np.isfinite(r): return 1e12
    r += -0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
    return -r

def run(job):
    tag,x0=job
    x0=np.clip(np.asarray(x0,float),B[:,0],B[:,1])
    r=minimize(target,x0,method="Powell",bounds=list(map(tuple,B)),
               options=dict(maxfev=60000,xtol=1e-4,ftol=1e-6))
    return tag,float(-r.fun),r.x.tolist()

jobs=[("branchA_best",best.copy())]
# branch-B seeds: XRT below nu_c  -> low eps_B, low n, higher p
for p_,eb,n_ in itertools.product([2.4,2.6,2.8],[1e-5,1e-4,1e-3],[1.0,10.0,100.0]):
    x=best.copy()
    x[IX["p"]]=p_; x[IX["log10_eps_B"]]=np.log10(eb); x[IX["log10_n_ism"]]=np.log10(n_)
    # compensate flux: raise E_iso and eps_e
    x[IX["log10_E_iso_core"]]=np.log10(3e52); x[IX["log10_eps_e"]]=np.log10(0.15)
    x[IX["log10_theta_c_core"]]=np.log10(0.05)
    jobs.append((f"branchB_p{p_}_eB{eb:g}_n{n_:g}",x))
print(f"{len(jobs)} starts")
with Pool(32) as pool: out=pool.map(run,jobs)
out.sort(key=lambda x:-x[1])
json.dump({"labels":labels,"B":B.tolist(),"results":out},
          open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/branch.json","w"))
print(f"branchA baseline target = {-target(best):.2f}\n")
for tag,v,x in out[:14]: print(f"{tag:<30} target={v:>10.2f}")
tag,v,x=out[0]; x=np.array(x)
print(f"\n=== BEST {tag}  target={v:.2f} ===")
print(f"{'label':<24}{'sampled':>10}{'physical':>14}")
for i,l in enumerate(labels):
    print(f"{l:<24}{x[i]:>10.4f}{(10**x[i] if LOGP[i] else x[i]):>14.5g}")
