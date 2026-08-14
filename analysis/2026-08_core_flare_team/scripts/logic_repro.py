import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index
xrt,opt=load_all_optical_data(); xi=load_xrt_spectral_index()
pd=make_param_defs(True,True)
R="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026"
s=np.load(R+"/samples.npy"); lp=np.load(R+"/log_probs.npy")
print("samples",s.shape,"stored best logP",np.nanmax(lp))
for tag,th in [("top_k[0]",np.load(R+"/top_k_params.npy")[0]),
               ("samples argmax",s.reshape(-1,s.shape[-1])[np.nanargmax(lp)])]:
    p={q.name:(10**v if q.scale is Scale.LOG else v) for q,v in zip(pd,th)}
    xm,om,si=compute_model_flux_all_bands(p,xrt,opt,True,True,xi)
    xc=np.sum(((xrt['flux']-xm)/xrt['flux_error'])**2)
    oc=sum(np.sum(((d['flux_mJy']-m)/d['flux_err'])**2) for d,m in zip(opt,om))
    r=(xrt['flux']-xm)/xrt['flux_error']; late=[f"{r[i]:+.1f}" for i in np.where(xrt['time']>1e5)[0]]
    print(f"  {tag:<16} logP={log_probability(th,pd,xrt,opt,True,True,xi):>9.2f} XRT={xc:7.1f} opt={oc:8.1f} SI={si:6.1f} lateXRT={late}")
# FLARE-X late XRT resid (likelihood-only, prior is -inf)
thf=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
pdf=make_param_defs(True,False)
p={q.name:(10**v if q.scale is Scale.LOG else v) for q,v in zip(pdf,thf)}
xm,om,si=compute_model_flux_all_bands(p,xrt,opt,True,False,xi)
r=(xrt['flux']-xm)/xrt['flux_error']
print(f"  {'FLARE-X':<16} lateXRT={[f'{r[i]:+.1f}' for i in np.where(xrt['time']>1e5)[0]]}")
