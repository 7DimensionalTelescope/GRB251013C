import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index
from multiprocessing import Pool

xrt_data, optical_datasets = fm.load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = fm.make_param_defs(True, True)
labels = [f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy"); lp=np.load(d+"/log_probs.npy")
best=s[np.argmax(lp)].copy()
print("baseline logL(+SI) =", fm.log_likelihood(best,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data))
print("n SI points:", len(xrt_index_data["time"]))

# 1-D scans, extending BEYOND current bounds
SCANS={
 "log10_theta_c_core": np.linspace(-1.7,-0.8,13),
 "log10_n_ism":        np.linspace(1.7,3.2,13),
 "p":                  np.linspace(2.05,2.85,13),
 "p_r":                np.linspace(2.2,3.6,13),
 "log10_eps_B_r":      np.linspace(-1.4,-0.1,13),
 "log10_eps_B":        np.linspace(-3.2,-1.6,13),
 "log10_Gamma0_core":  np.linspace(2.5,3.3,13),
 "log10_E_iso_wing":   np.linspace(50.8,52.6,13),
 "log10_eps_e_wing":   np.linspace(-1.8,0.0,13),
 "p_wing":             np.linspace(2.4,3.5,13),
 "log10_theta_c_wing": np.linspace(-0.9,-0.05,13),
 "xi_wing":            np.linspace(0.5,1.0,11),
}
def ev(args):
    i,v=args
    th=best.copy(); th[i]=v
    return fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
jobs=[]; meta=[]
for name,grid in SCANS.items():
    i=labels.index(name)
    for v in grid: jobs.append((i,v)); meta.append((name,v))
with Pool(48) as pool: res=pool.map(ev,jobs)
out={}
for (name,v),r in zip(meta,res): out.setdefault(name,[]).append((float(v),float(r)))
json.dump(out,open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/scan.json","w"))
B={l:(np.log10(p.lower) if p.scale is fm.Scale.LOG else p.lower,
      np.log10(p.upper) if p.scale is fm.Scale.LOG else p.upper) for l,p in zip(labels,pdefs)}
base=fm.log_likelihood(best,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
for name,vals in out.items():
    lo,hi=B[name]
    print("\n"+name+f"   [current bounds {lo:.3f} .. {hi:.3f}]  best={best[labels.index(name)]:.3f}")
    bestv=max(vals,key=lambda x:x[1])
    for v,r in vals:
        mark="*" if (v,r)==bestv else " "
        inb="in " if lo<=v<=hi else "OUT"
        print(f"   {v:>8.3f} {inb}  dlogL={r-base:>9.1f} {mark}")
