import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys,numpy as np,json
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C")
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index,HOST_AV_LOG10_MEAN,HOST_AV_LOG10_SIGMA
from scipy.optimize import minimize
from multiprocessing import Pool
xrt_data,od=fm.load_all_optical_data(); sid=load_xrt_spectral_index()
pdefs=fm.make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}; LOGP=[p.scale is fm.Scale.LOG for p in pdefs]
d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy");lp=np.load(d+"/log_probs.npy");best=s[np.argmax(lp)].copy()
iav=IX["log10_A_V"]
def tgt(th):
    r=fm.log_likelihood(th,pdefs,xrt_data,od,True,True,sid)
    if not np.isfinite(r): return -1e12
    return r-0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
BASE=tgt(best)
# free ONLY the flux-normalisation trio; hold the Gamma-matching (p, eps_B, n_ism) fixed
FREE=["log10_E_iso_core","log10_eps_e","log10_theta_c_core"]
bb=np.array([[np.log10(1e51),np.log10(1e54)],
             [np.log10(0.005),np.log10(0.5)],
             [np.log10(0.005),np.log10(0.3)]])
idx=[IX[f] for f in FREE]
def run(job):
    tag,p_,eb,n_=job
    x0=best.copy()
    if p_ is not None:
        x0[IX["p"]]=p_; x0[IX["log10_eps_B"]]=np.log10(eb); x0[IX["log10_n_ism"]]=np.log10(n_)
    def f(y):
        th=x0.copy(); th[idx]=np.clip(y,bb[:,0],bb[:,1]); return -tgt(th)
    r=minimize(f,x0[idx].copy(),method="Powell",bounds=list(map(tuple,bb)),
               options=dict(maxfev=1500,xtol=1e-3,ftol=1e-6))
    th=x0.copy(); th[idx]=r.x
    return tag,float(-r.fun),th.tolist()
jobs=[("A_baseline_renorm",None,None,None),
      ("B_p2.2_eB1e-4_n137",2.2,1e-4,137.0),
      ("B_p2.4_eB1e-4_n10",2.4,1e-4,10.0),
      ("B_p2.4_eB1e-5_n137",2.4,1e-5,137.0),
      ("B_p2.6_eB1e-5_n30",2.6,1e-5,30.0),
      ("B_p2.6_eB1e-6_n137",2.6,1e-6,137.0)]
with Pool(6) as pool: out=pool.map(run,jobs)
out.sort(key=lambda x:-x[1])
print(f"BASELINE (171919 best, untouched) logL+prior = {BASE:.2f}\n")
for t,v,x in out: print(f"{t:<24} logL+prior={v:>10.2f}   delta={v-BASE:>+9.2f}")
json.dump({"labels":labels,"baseline":BASE,"results":out},
          open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/decide.json","w"))
t,v,x=out[0]; x=np.array(x)
print(f"\n=== BEST: {t}  logL+prior={v:.2f} (delta {v-BASE:+.2f}) ===")
for i,l in enumerate(labels):
    ch=" <--" if abs(x[i]-best[i])>1e-6 else ""
    print(f"  {l:<24}{x[i]:>10.4f}{(10**x[i] if LOGP[i] else x[i]):>14.5g}{ch}")
