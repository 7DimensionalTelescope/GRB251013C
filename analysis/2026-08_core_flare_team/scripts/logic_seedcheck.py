import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.params import make_param_defs, ParamDefWithPrior
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability
from grb.spectral_index import load_xrt_spectral_index
xrt,opt=load_all_optical_data(); xi=load_xrt_spectral_index()
pdefs=make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}; LOGP=[p.scale is Scale.LOG for p in pdefs]
def widen(n,lo,hi):
    for i,p in enumerate(pdefs):
        if p.name==n: pdefs[i]=ParamDefWithPrior(n,lo,hi,p.scale,gaussian_prior=p.gaussian_prior)
WIDE={"E_iso_core":(1e51,1e55),"Gamma0_core":(100,2000),"theta_c_core":(0.02,0.8),
 "n_ism":(0.01,3000),"p":(2.01,3.2),"eps_e":(0.001,0.5),"eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "tau":(5,100),"p_r":(2.0,3.5),"eps_e_r":(0.005,0.5),"eps_B_r":(1e-4,0.6),"xi_r":(0.3,1.0),
 "t_start_flare":(500,2e4),"tau_rise_flare":(5,5000),"tau_decay_flare":(500,1e5),
 "A_flare":(1e-11,2e-8),"flare_beta":(0.0,2.0),
 "E_iso_wing":(1e50,1e53),"Gamma0_wing":(3,100),"theta_c_wing":(0.2,1.0),
 "p_wing":(2.01,3.3),"eps_e_wing":(0.005,1.0),"eps_B_wing":(1e-6,0.1),"xi_wing":(0.1,1.0)}
for k,(lo,hi) in WIDE.items(): widen(k,lo,hi)
INC=np.load("/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026/top_k_params.npy")[0]
FX=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
def hybrid(w):
    x=INC.copy(); x[:19]=FX; x[IX["p"]]=max(FX[IX["p"]],2.02)
    for k,v in w.items():
        i=IX.get("log10_"+k, IX.get(k)); x[i]=np.log10(v) if LOGP[i] else v
    return x
HYB_A=hybrid(dict(E_iso_wing=2e51,Gamma0_wing=8.0,theta_c_wing=0.5,p_wing=2.5,eps_e_wing=0.1,eps_B_wing=1e-5,xi_wing=0.5))
HYB_B=hybrid({})
INC_W=INC.copy(); INC_W[IX["log10_theta_c_core"]]=np.log10(0.15); INC_W[IX["log10_Gamma0_core"]]=np.log10(300)
# the seed that is MISSING: FLARE-X core + wing driven to the E_iso floor
HYB_NULL=hybrid(dict(E_iso_wing=1.5e50,Gamma0_wing=8.0,theta_c_wing=0.5,p_wing=2.5,eps_e_wing=0.1,eps_B_wing=1e-5,xi_wing=0.5))
for tag,th in [("INC (20 walkers)",INC),("HYB_A (16)",HYB_A),("HYB_B (14)",HYB_B),
               ("INC_W (14)",INC_W),("HYB_NULL (0 -- MISSING)",HYB_NULL)]:
    bad=[labels[i] for i,(p,v) in enumerate(zip(pdefs,th))
         if not ((np.log10(p.lower) if LOGP[i] else p.lower)<=v<=(np.log10(p.upper) if LOGP[i] else p.upper))]
    print(f"{tag:<24} logP={log_probability(th,pdefs,xrt,opt,True,True,xi):>10.1f}  out_of_box={bad}")
