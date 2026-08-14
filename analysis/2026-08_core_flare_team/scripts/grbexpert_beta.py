import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np, sys
sys.path.insert(0,'/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor')
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import make_core_model
from grb.likelihood import spectral_index_model
from grb.utils import model_array
from grb.const import XRT_BAND
from grb.functions import norris_flare
from grb.extinction import host_extinction_attenuation
from grb.const import REDSHIFT
v=np.load('/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy')
pds=make_param_defs(True,False)
P={pd.name:(10**x if pd.scale is Scale.LOG else x) for pd,x in zip(pds,v)}

print("=== XRT model spectral slope beta (Gamma=1-beta... i.e. PhotonIndex=1+|beta|) vs n_ism ===")
print("   observed: Gamma_ph = 1.82 +/- 0.2  ->  beta_obs = -0.82")
ts=np.array([1e3,1e4,1e5,4e5])
print(f"{'n_ism':>8} " + " ".join(f"{'G@%.0e'%t:>10}" for t in ts))
for n in [0.3,1,3,10,30,134]:
    Q=dict(P); Q['n_ism']=n
    m=make_core_model(Q)
    b,_=spectral_index_model(m,None,Q,ts,False)
    print(f"{n:8.4g} " + " ".join(f"{1-bb:10.3f}" for bb in b))

print("\n=== Flare contribution to the OPTICAL at the 7DT epoch (6.47 hr, r-band 4.84e14 Hz) ===")
t=np.array([6.47*3600.]); nu=4.84e14
m=make_core_model(P)
syn=float(np.atleast_1d(model_array(m.flux_density(t,np.array([nu])).total))[0])
ft=norris_flare(t,P["t_start_flare"],P["tau_rise_flare"],P["tau_decay_flare"],P["A_flare"])
bF=P["flare_beta"]; lo,hi=XRT_BAND[0],XRT_BAND[1]
from grb.const import XRT_NU_LO, XRT_NU_HI
K=ft*(1-bF)/(XRT_NU_HI**(1-bF)-XRT_NU_LO**(1-bF))
fl=float(np.atleast_1d(K*nu**(-bF))[0])
att=host_extinction_attenuation(nu,P["A_V"],REDSHIFT)
print(f"  synchrotron (FS+RS) = {syn*1e26*att:9.4g} mJy")
print(f"  flare extrapolation = {fl*1e26*att:9.4g} mJy")
print(f"  flare fraction of optical = {fl/(syn+fl):.3f}")
print(f"  optical/X-ray extrapolation factor over {np.log10(XRT_NU_LO/nu):.2f} dex at beta={bF:.2f}")
# flare fraction in XRT at its own peak
tp=P["t_start_flare"]+np.sqrt(P["tau_rise_flare"]*P["tau_decay_flare"])
tpa=np.array([tp])
xf=float(np.atleast_1d(model_array(m.flux(tpa,XRT_BAND[0],XRT_BAND[1],10).total))[0])
flp=float(np.atleast_1d(norris_flare(tpa,P["t_start_flare"],P["tau_rise_flare"],P["tau_decay_flare"],P["A_flare"]))[0])
print(f"\n  flare peak t={tp:.0f}s : XRT afterglow={xf:.3g}, flare={flp:.3g}, flare frac={flp/(flp+xf):.2f}")
print(f"  flare width dt/t = tau_decay/t_peak = {P['tau_decay_flare']/tp:.2f}  (typical XRT flares: 0.1-0.5)")
