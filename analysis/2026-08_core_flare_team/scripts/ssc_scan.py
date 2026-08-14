"""Where can SSC dominate the late XRT band, and with what photon index?

Grid over (eps_B, n_ism, p, theta_c, E_iso): compute core-only XRT band flux
with ssc on/off, the local photon index across the XRT band, and the i-band
optical flux, at 38 and 113 hr. A viable corner needs:
  - late XRT flux boost (ssc/sync ratio >> 1) toward the data
    (4.1e-13 @38hr, 1.1e-13 @113hr)
  - photon index ~ 1.8 at those epochs
  - little added optical
Also prints early-time (0.15-1 hr) XRT so we see if the same core stays
consistent with the early data (~1e-10 @0.15hr).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, itertools, time
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet
from grb.const import REDSHIFT, D_L, XRT_BAND, XRT_NU_LO, XRT_NU_HI
from grb.utils import model_array

def build(E,G0,th,n,p,ee,eB,xi,ssc):
    jet=TophatJet(theta_c=th,E_iso=E,Gamma0=G0)
    return Model(jet=jet,medium=ISM(n_ism=n),
                 observer=Observer(lumi_dist=D_L,z=REDSHIFT,theta_obs=0.0),
                 fwd_rad=Radiation(eps_e=ee,eps_B=eB,p=p,xi_e=xi,ssc=ssc))

t_late=np.array([38.,113.])*3600
t_early=np.array([0.15,1.0])*3600
nu_i=3.93e14

def gamma_band(m,tt):
    lo=model_array(m.flux_density(tt,XRT_NU_LO*np.ones_like(tt)).total)
    hi=model_array(m.flux_density(tt,XRT_NU_HI*np.ones_like(tt)).total)
    with np.errstate(all="ignore"):
        return 1-np.log(hi/lo)/np.log(XRT_NU_HI/XRT_NU_LO)

t0=time.time()
rows=[]
GRID=list(itertools.product(
    [1e52,1e53],          # E_iso
    [0.1,0.3],            # theta_c
    [1.0,50.0,500.0],     # n_ism
    [2.2,2.6],            # p
    [1e-6,1e-5,1e-4,1e-3],# eps_B
))
print(f"{len(GRID)} configs; eps_e=0.1, xi=0.9, Gamma0=400 fixed")
print(f"{'E':>6} {'th':>4} {'n':>5} {'p':>4} {'eB':>7} | {'F38(ssc)':>9} {'boost':>6} {'G38':>5} {'G113':>5} | {'F0.15h':>9} {'i38/uJy':>8}")
for E,th,n,p,eB in GRID:
    try:
        m1=build(E,400,th,n,p,0.1,eB,0.9,True)
        m0=build(E,400,th,n,p,0.1,eB,0.9,False)
        f1=model_array(m1.flux(t_late,XRT_BAND[0],XRT_BAND[1],10).total)
        f0=model_array(m0.flux(t_late,XRT_BAND[0],XRT_BAND[1],10).total)
        fe=model_array(m1.flux(t_early,XRT_BAND[0],XRT_BAND[1],10).total)
        g=gamma_band(m1,t_late)
        oi=model_array(m1.flux_density(t_late,nu_i*np.ones_like(t_late)).total)*1e26
        rows.append((E,th,n,p,eB,f1,f0,g,fe,oi))
        print(f"{E:6.0e} {th:4.1f} {n:5.0f} {p:4.1f} {eB:7.0e} | {f1[0]:9.2e} {f1[0]/f0[0]:6.1f} {g[0]:5.2f} {g[1]:5.2f} | {fe[0]:9.2e} {oi[0]*1e3:8.2f}",flush=True)
    except Exception as e:
        print(f"{E:6.0e} {th:4.1f} {n:5.0f} {p:4.1f} {eB:7.0e} | FAIL {type(e).__name__}: {e}")
print(f"\ntargets: F38=4.1e-13 F113=1.1e-13 Gamma=1.82+-0.21; early F(0.15h)=1.1e-10")
print(f"i-band data @38hr: none (last opt ~10hr, ~0.5 mJy at 5hr); wing supplies opt")
print(f"{time.time()-t0:.0f}s total")
