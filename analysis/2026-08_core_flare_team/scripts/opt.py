import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index
from scipy.optimize import minimize
from multiprocessing import Pool

xrt_data, optical_datasets = fm.load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = fm.make_param_defs(True, True)
labels = [f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
CUR = np.array([[np.log10(p.lower) if p.scale is fm.Scale.LOG else p.lower,
                 np.log10(p.upper) if p.scale is fm.Scale.LOG else p.upper] for p in pdefs])
# proposed widened bounds (sampled space)
WIDE = CUR.copy()
def setb(name, lo=None, hi=None, log=True):
    i=labels.index(name)
    if lo is not None: WIDE[i,0]=np.log10(lo) if log else lo
    if hi is not None: WIDE[i,1]=np.log10(hi) if log else hi
setb("log10_theta_c_core", hi=0.10)
setb("log10_n_ism",        hi=600)
setb("log10_eps_B",        lo=5e-4)
setb("log10_Gamma0_core",  hi=1600)
setb("p",     lo=2.01, hi=2.6, log=False)
setb("p_r",   lo=2.0,  hi=3.6, log=False)
setb("log10_eps_B_r",      hi=0.6)
setb("log10_E_iso_wing",   lo=5e50, hi=2e53)
setb("log10_eps_e_wing",   lo=0.01, hi=1.0)
setb("p_wing", lo=2.1, hi=3.4, log=False)
setb("log10_theta_c_wing", lo=0.05, hi=0.9)
setb("log10_tau", lo=2, hi=60)

d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy"); lp=np.load(d+"/log_probs.npy")
best=s[np.argmax(lp)].copy()

def nll(th, B):
    if np.any(th<B[:,0]) or np.any(th>B[:,1]): return 1e12
    r = fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    return -r if np.isfinite(r) else 1e12

def run(job):
    tag, B, seed, scale = job
    rng=np.random.default_rng(seed)
    x0=best.copy()
    if scale>0:
        x0 = x0 + rng.normal(0,scale,len(x0))*(B[:,1]-B[:,0])
    x0=np.clip(x0,B[:,0],B[:,1])
    res=minimize(nll,x0,args=(B,),method="Powell",
                 bounds=list(map(tuple,B)),
                 options=dict(maxfev=40000,xtol=1e-4,ftol=1e-6))
    return tag, float(-res.fun), res.x.tolist()

jobs=[]
for k in range(6):
    jobs.append((f"CUR_s{k}", CUR, 100+k, 0.0 if k==0 else 0.01*k))
for k in range(10):
    jobs.append((f"WIDE_s{k}", WIDE, 200+k, 0.0 if k==0 else 0.01*k))
with Pool(16) as pool: out=pool.map(run,jobs)
out.sort(key=lambda x:-x[1])
print(f"baseline logL = {fm.log_likelihood(best,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data):.2f}\n")
for tag,v,x in out: print(f"{tag:<12} logL={v:>10.2f}")
json.dump({"labels":labels,"CUR":CUR.tolist(),"WIDE":WIDE.tolist(),
           "results":[[t,v,x] for t,v,x in out]},
          open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/opt.json","w"))
bt,bv,bx=out[0]
print(f"\n=== best overall: {bt}  logL={bv:.2f} ===")
print(f"{'label':<24}{'value':>10}{'phys':>14}   {'CURlo':>8}{'CURhi':>8}  at_cur_bound")
for i,l in enumerate(labels):
    ph = 10**bx[i] if pdefs[i].scale is fm.Scale.LOG else bx[i]
    hit = "" 
    if bx[i] <= CUR[i,0]+1e-6*(CUR[i,1]-CUR[i,0]): hit="LOW"
    if bx[i] >= CUR[i,1]-1e-6*(CUR[i,1]-CUR[i,0]): hit="HIGH"
    out_of = "OUTSIDE_CUR" if (bx[i]<CUR[i,0]-1e-9 or bx[i]>CUR[i,1]+1e-9) else ""
    print(f"{l:<24}{bx[i]:>10.4f}{ph:>14.4g}   {CUR[i,0]:>8.3f}{CUR[i,1]:>8.3f}  {hit}{out_of}")
