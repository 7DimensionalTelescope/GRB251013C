import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index

xrt_data, optical_datasets = fm.load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = fm.make_param_defs(True, True)
labels = [f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]

base="/data/dtak/research/grb/GRB251013C/modeling/fit_results"
runs=sorted(d for d in os.listdir(base) if d.startswith("final_flare_wing_2026"))
cands=[]
for r in runs:
    d=os.path.join(base,r)
    try:
        s=np.load(os.path.join(d,"samples.npy")); lp=np.load(os.path.join(d,"log_probs.npy"))
        lbl=open(os.path.join(d,"labels.txt")).read().split()
        assert lbl==labels, r
        if not np.isfinite(lp).any(): continue
        cands.append((r,"best", s[np.argmax(lp)]))
        tk=np.load(os.path.join(d,"top_k_params.npy"))
        for i in range(min(5,len(tk))): cands.append((r,f"top{i}",tk[i]))
    except Exception as e:
        print("skip",r,e)

print(f"{len(cands)} candidates\n")
rows=[]
for r,tag,th in cands:
    ll_si = fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
    ll_no = fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,None)
    lpri  = fm.log_prior(th,pdefs)   # -inf if outside CURRENT bounds
    rows.append((r,tag,ll_si,ll_no,np.isfinite(lpri),th))
rows.sort(key=lambda x:-x[2])
print(f"{'run':<34}{'tag':<6}{'logL(+SI)':>12}{'logL(noSI)':>12}{'SIchi2':>9}  in_bounds")
for r,tag,a,b,ok,th in rows:
    print(f"{r:<34}{tag:<6}{a:>12.1f}{b:>12.1f}{-2*(a-b):>9.1f}  {ok}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/ranked.npy", np.array([r[5] for r in rows]))
import json
json.dump([[r[0],r[1],r[2],r[3],bool(r[4])] for r in rows], open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/ranked.json","w"))
json.dump(labels, open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/labels.json","w"))
