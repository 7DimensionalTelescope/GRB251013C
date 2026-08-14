import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import final_model as fm
from utils import load_xrt_spectral_index
from multiprocessing import Pool
xrt_index_data = load_xrt_spectral_index()
pdefs = fm.make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
d="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260724_171919"
s=np.load(d+"/samples.npy"); lp=np.load(d+"/log_probs.npy"); best=s[np.argmax(lp)].copy()
# observed mean photon index (pre-flare, t<2000s)
t=xrt_index_data["time"]; beta=xrt_index_data["beta"]
m=t<2000
gam=1-beta
print(f"observed photon index (t<2000s): mean={gam[m].mean():.3f}  range {gam[m].min():.2f}-{gam[m].max():.2f}  n={m.sum()}")
def core_gamma(job):
    p_,eb,n_=job
    th=best.copy()
    th[IX["p"]]=p_; th[IX["log10_eps_B"]]=np.log10(eb); th[IX["log10_n_ism"]]=np.log10(n_)
    params={}
    for pd,v in zip(pdefs,th):
        params[pd.name]=10**v if pd.scale is fm.Scale.LOG else v
    core=fm.make_core_model(params)
    bm,keep=fm.spectral_index_model(core,None,params,np.array([500.0,1500.0]),False)
    return p_,eb,n_,1-bm[0],1-bm[1]
jobs=[(p_,eb,n_) for p_ in (2.2,2.4,2.6,2.8) for eb in (5e-3,1e-3,1e-4,1e-5,1e-6) for n_ in (137.0,10.0)]
with Pool(20) as pool: res=pool.map(core_gamma,jobs)
print(f"\n{'p':>5}{'eps_B':>9}{'n_ism':>8}{'Gam(500s)':>11}{'Gam(1500s)':>12}   note")
for p_,eb,n_,g1,g2 in res:
    note=""
    if abs(g1-1.85)<0.08: note="<== matches data"
    print(f"{p_:>5.2f}{eb:>9.0e}{n_:>8.0f}{g1:>11.3f}{g2:>12.3f}   {note}")
