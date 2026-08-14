import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np, sys
sys.path.insert(0,'/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor')
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import make_core_model, load_all_optical_data
from grb.utils import model_array
from grb.const import XRT_BAND

v=np.load('/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy')
pds=make_param_defs(True,False)
P={}
for pd,x in zip(pds,v):
    P[pd.name]=10**x if pd.scale is Scale.LOG else x
# rename to model keys
P['E_iso_core']=P['E_iso_core']; 
m=make_core_model(P)

print("=== RADIO PREDICTIONS (core FS+RS, no flare, no host ext) ===")
obs=[("ALMA  97.5GHz", 5.5*3600., 97.5e9, 0.30, None),
     ("AMI   15.5GHz", 27.79*3600., 15.5e9, 0.34, 0.08),
     ("VLA   10.0GHz", 1.36*86400., 10.0e9, 0.111, 0.012)]
for name,t,nu,fo,fe in obs:
    ts=np.array([t]); r=m.flux_density(ts, np.array([nu]))
    tot=float(np.atleast_1d(model_array(r.total))[0])*1e26
    print(f"{name}  t={t:9.3g}s  model={tot:10.4g} mJy   obs={fo} +/- {fe}   model/obs={tot/fo:8.3g}")

print()
print("=== RADIO SED at t=1.2 d (locate nu_a) ===")
t=1.2*86400.
nus=np.logspace(9, 12.2, 17)
ts=np.full_like(nus,t)
f=model_array(m.flux_density(ts,nus).total)*1e26
for nu,fl in zip(nus,f):
    print(f"  nu={nu/1e9:9.3f} GHz   F={fl:11.4g} mJy")
sl=np.gradient(np.log(f),np.log(nus))
print("  log-slopes:", np.round(sl,2))

print()
print("=== XRT-BAND LIGHT CURVE to 40 d: where is the jet break? ===")
tt=np.logspace(np.log10(3e4), np.log10(40*86400.), 25)
fx=model_array(m.flux(tt, XRT_BAND[0], XRT_BAND[1], 10).total)
sl=np.gradient(np.log(fx),np.log(tt))
for t_,f_,s_ in zip(tt,fx,sl):
    print(f"  t={t_/86400.:8.3f} d  F={f_:11.4g}   dlnF/dlnt={s_:6.2f}")
