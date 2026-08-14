import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np, sys
sys.path.insert(0,'/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor')
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import make_core_model, load_all_optical_data
from grb.utils import model_array
from grb.const import XRT_BAND, D_L, REDSHIFT

v=np.load('/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy')
pds=make_param_defs(True,False)
P={pd.name:(10**x if pd.scale is Scale.LOG else x) for pd,x in zip(pds,v)}

xrt,opt=load_all_optical_data()
m=make_core_model(P)
fx=model_array(m.flux(xrt['time'],XRT_BAND[0],XRT_BAND[1],10).total)
print("=== UNIT CHECK: model vs observed XRT (last 6 pts) ===")
for t,fo,fm in list(zip(xrt['time'],xrt['flux'],fx))[-6:]:
    print(f"  t={t/86400.:8.3f} d  obs={fo:11.4g}  model={fm:11.4g}  ratio={fm/fo:6.2f}")

print()
print("=== CHANDRA EPOCH (27.396 d), obs = 2.5e-15 erg/cm2/s (0.3-10 keV) ===")
tc=np.array([27.396*86400.])
fc=float(np.atleast_1d(model_array(m.flux(tc,XRT_BAND[0],XRT_BAND[1],10).total))[0])
print(f"  FLARE-X model = {fc:.4g}   obs = 2.5e-15   model/obs = {fc/2.5e-15:.2f}")

print()
print("=== ALMA 97.5 GHz @5.5hr: scan n_ism (all else fixed) ; obs ~0.30 mJy ===")
t_alma=np.array([5.5*3600.]); nu_alma=np.array([97.5e9])
t_ami=np.array([27.79*3600.]); t_vla=np.array([1.36*86400.])
print(f"{'n_ism':>8} {'ALMA(mJy)':>11} {'AMI15.5':>10} {'VLA10':>10} {'XRT@4.5d':>11}")
for n in [1,3,10,30,50,100,133.7,300]:
    Q=dict(P); Q['n_ism']=n
    mm=make_core_model(Q)
    a=float(np.atleast_1d(model_array(mm.flux_density(t_alma,nu_alma).total))[0])*1e26
    b=float(np.atleast_1d(model_array(mm.flux_density(t_ami,np.array([15.5e9])).total))[0])*1e26
    c=float(np.atleast_1d(model_array(mm.flux_density(t_vla,np.array([10e9])).total))[0])*1e26
    x=float(np.atleast_1d(model_array(mm.flux(np.array([4.5*86400.]),XRT_BAND[0],XRT_BAND[1],10).total))[0])
    print(f"{n:8.4g} {a:11.4g} {b:10.4g} {c:10.4g} {x:11.4g}")
print("  observed:            0.30       0.34+-0.08  0.111+-0.012")
