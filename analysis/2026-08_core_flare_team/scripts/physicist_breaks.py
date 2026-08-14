import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
import sys
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np
from grb.modeling import make_core_model

NU_I, NU_X = 3.932e14, 1.2e17

def report(over, tag):
    BASE = dict(E_iso_core=1.08e52, Gamma0_core=136.0, theta_c_core=0.760, n_ism=134.0,
                p=2.121, eps_e=0.034, eps_B=0.030, xi=0.31155838, tau=46.5,
                p_r=2.7705, eps_e_r=0.1084, eps_B_r=0.528, xi_r=0.88184)
    BASE.update(over)
    m = make_core_model(BASE); d = m.details(20, 1e6)
    f, r = d.fwd, d.rvs
    tobs = np.asarray(f.t_obs)   # (phi,theta,t)?
    print(f"\n===== {tag}   t_obs shape {tobs.shape}")
    # on-axis slice: theta index 0
    idx = (0,0)
    t = tobs[idx]
    num = np.asarray(f.nu_m)[idx]; nuc = np.asarray(f.nu_c)[idx]
    G   = np.asarray(f.Gamma)[idx]; B = np.asarray(f.B_comv)[idx]
    rt  = np.asarray(r.t_obs)[idx]; rnum = np.asarray(r.nu_m)[idx]; rnuc = np.asarray(r.nu_c)[idx]
    rGam= np.asarray(r.Gamma)[idx]; rN = np.asarray(r.N_e)[idx]
    print(" FWD:  t_obs      Gamma      nu_m        nu_c     regime@i-band")
    for tt in [50,100,200,500,1000,2000,5000,2e4,1e5,4e5]:
        nm = np.interp(tt,t,num); nc = np.interp(tt,t,nuc); gg=np.interp(tt,t,G)
        reg = "nu<nu_m" if NU_I<min(nm,nc) else ("nu>both" if NU_I>max(nm,nc) else ("nu_m<nu<nu_c" if nm<nc else "nu_c<nu<nu_m(fast)"))
        regX = "X>both" if NU_X>max(nm,nc) else "X between"
        print(f"   {tt:9.0f}  {gg:8.2f}  {nm:10.3e} {nc:10.3e}   {reg:20s} | {regX}")
    print(" RVS:  t_obs     Gamma      nu_m        nu_c        N_e")
    for tt in [20,50,100,200,500,1000,2000]:
        if tt < rt.min() or tt > rt.max():
            print(f"   {tt:9.0f}   -- outside RS grid ({rt.min():.1f}-{rt.max():.3g})"); continue
        print(f"   {tt:9.0f}  {np.interp(tt,rt,rGam):8.2f}  {np.interp(tt,rt,rnum):10.3e} {np.interp(tt,rt,rnuc):10.3e}  {np.interp(tt,rt,rN):10.3e}")

report({}, "FLARE-X core (n=134, eps_B=0.030, xi=0.31)")
report(dict(n_ism=3.0, eps_B=0.005), "low-density n=3 eps_B=0.005")
