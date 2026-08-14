import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np, sys
sys.path.insert(0,'/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor')
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import make_core_model
from grb.utils import model_array
v=np.load('/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy')
pds=make_param_defs(True,False)
P={pd.name:(10**x if pd.scale is Scale.LOG else x) for pd,x in zip(pds,v)}
Pf=dict(P)
for k in ("p_r","eps_e_r","eps_B_r"): Pf.pop(k,None)   # FS only
m_tot=make_core_model(P); m_fs=make_core_model(Pf)
print("=== FS vs FS+RS at the radio/mm epochs (mJy) ===")
for nm,t,nu,obs in [("ALMA 97.5GHz",5.5*3600.,97.5e9,0.30),
                    ("AMI  15.5GHz",27.79*3600.,15.5e9,0.34),
                    ("VLA  10.0GHz",1.36*86400.,10.0e9,0.111)]:
    ts=np.array([t]); ns=np.array([nu])
    a=float(np.atleast_1d(model_array(m_tot.flux_density(ts,ns).total))[0])*1e26
    b=float(np.atleast_1d(model_array(m_fs.flux_density(ts,ns).total))[0])*1e26
    print(f"  {nm}  FS+RS={a:9.4g}  FS only={b:9.4g}  obs={obs}   RS frac={1-b/a:5.2f}")

# ---- SN 1998bw scaled to z=0.572 ----
from astropy.cosmology import Planck18
import astropy.units as u
z=0.572
DL=Planck18.luminosity_distance(z).to(u.cm).value
DM=5*np.log10(Planck18.luminosity_distance(z).to(u.pc).value/10)
print(f"\n=== SN template at z={z} ===")
print(f"  D_L = {DL:.4g} cm ; distance modulus = {DM:.3f} mag")
print(f"  observed R/r (641/620 nm) samples rest-frame {641/(1+z):.0f}/{620/(1+z):.0f} nm  (~B band)")
K=-2.5*np.log10(1+z)
print(f"  matched-filter K-correction K = -2.5log10(1+z) = {K:.3f}")
for nm,MB,tpk in [("SN1998bw (M_B~-18.7)",-18.7,14.),("SN1998bw (M_V~-19.3)",-19.3,16.),
                  ("SN2006aj (M_B~-18.3)",-18.3,9.),("SN2011kl (M~-20.0)",-20.0,14.)]:
    print(f"  {nm:24s} -> obs mag = {MB+DM+K:6.2f} AB ; peak at {tpk*(1+z):5.1f} d obs = {tpk*(1+z)*86400:.2e} s")
print(f"\n  OBSERVED day 12-13 bump: R=21.70 Vega ~ {21.70+0.21:.2f} AB  -> M = {21.91-DM-K:.2f}")
