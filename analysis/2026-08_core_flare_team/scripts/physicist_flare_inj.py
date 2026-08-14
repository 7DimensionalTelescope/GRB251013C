import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
import sys
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np
from grb.modeling import make_core_model
from grb.functions import norris_flare
from grb.utils import model_array
from grb.extinction import host_extinction_attenuation
from grb.const import REDSHIFT, XRT_BAND, XRT_NU_LO, XRT_NU_HI

BASE = dict(E_iso_core=1.08e52, Gamma0_core=136.0, theta_c_core=0.760, n_ism=134.0,
            p=2.121, eps_e=0.034, eps_B=0.030, xi=0.31155838, tau=46.5,
            p_r=2.7705, eps_e_r=0.1084, eps_B_r=0.528, xi_r=0.88184)
FL = dict(t_start_flare=2087., tau_rise_flare=98.6, tau_decay_flare=7447., A_flare=4.72e-10,
          flare_beta=0.68325)
AV = 0.03585; NU_I = 3.932e14
print("XRT_BAND", XRT_BAND, "NU_LO/HI", XRT_NU_LO, XRT_NU_HI)
print("Norris peak at t-t0 = sqrt(tr*td) = %.0f s -> t_peak = %.0f s"%(
    np.sqrt(FL["tau_rise_flare"]*FL["tau_decay_flare"]), FL["t_start_flare"]+np.sqrt(FL["tau_rise_flare"]*FL["tau_decay_flare"])))

m = make_core_model(BASE)
tg = np.logspace(np.log10(80), np.log10(6e5), 200)
xrt_core = model_array(m.flux(tg, XRT_BAND[0], XRT_BAND[1], 10).total)
fl = norris_flare(tg, FL["t_start_flare"], FL["tau_rise_flare"], FL["tau_decay_flare"], FL["A_flare"])
att = host_extinction_attenuation(NU_I, AV, REDSHIFT)
opt_core = model_array(m.flux_density(tg, NU_I*np.ones_like(tg)).total)*1e26
b = FL["flare_beta"]
K = fl*(1-b)/(XRT_NU_HI**(1-b)-XRT_NU_LO**(1-b))
opt_flare = K*NU_I**(-b)*1e26
opt_tot = (opt_core+opt_flare)*att

# model photon index of core (synchrotron slope over XRT band)
flo = model_array(m.flux_density(tg, XRT_NU_LO*np.ones_like(tg)).total)
fhi = model_array(m.flux_density(tg, XRT_NU_HI*np.ones_like(tg)).total)
beta_core = np.log(fhi/flo)/np.log(XRT_NU_HI/XRT_NU_LO)
# blended photon index: nu F_nu weighted; approximate by combining nu-space densities
flo_fl = K*XRT_NU_LO**(-b); fhi_fl = K*XRT_NU_HI**(-b)
beta_tot = np.log((fhi+fhi_fl)/(flo+flo_fl))/np.log(XRT_NU_HI/XRT_NU_LO)

print("\n  t[s]    XRT_core     XRT_flare  flarefrac_X | i_core  i_flare  flarefrac_opt | Gam_core Gam_tot")
for tt in [1e3,2.5e3,5e3,1e4,2.34e4,4e4,7e4,1.36e5,4.06e5]:
    i = np.argmin(abs(tg-tt))
    fx=fl[i]/(fl[i]+xrt_core[i]); fo=opt_flare[i]/(opt_flare[i]+opt_core[i])
    print(f"{tg[i]:9.3g} {xrt_core[i]:11.3e} {fl[i]:11.3e} {fx:8.3f}  | {opt_core[i]*att:7.4f} {opt_flare[i]*att:8.4f} {fo:8.3f} | "
          f"{1-beta_core[i]:8.3f} {1-beta_tot[i]:7.3f}")

def slope(t,y,lo,hi):
    k=(t>=lo)&(t<=hi)&(y>0)
    return -np.polyfit(np.log10(t[k]),np.log10(y[k]),1)[0]
print("\nMODEL slopes (core only, no flare):")
print("  alpha_X(454-1900)  = %.3f   [obs 1.099+-0.095]"%slope(tg,xrt_core,454,1900))
print("  alpha_i(454-1900)  = %.3f   [obs 0.640+-0.077]"%slope(tg,opt_core,454,1900))
print("  alpha_i(94-2087)   = %.3f   [obs 0.573+-0.020]"%slope(tg,opt_core,94,2087))
print("MODEL slopes (core+flare):")
print("  alpha_i(454-1900)  = %.3f"%slope(tg,opt_tot,454,1900))
print("  alpha_X(1e5-4e5)   = %.3f"%slope(tg,xrt_core+fl,1e5,4e5))
print("  alpha_i(1e5-4e5)   = %.3f"%slope(tg,opt_tot,1e5,4e5))

print("\n=== energy-injection budget for the 2-3e5 s optical bump ===")
for fac in [1.5,2.0,2.5,3.0]:
    p2=dict(BASE); p2["E_iso_core"]=BASE["E_iso_core"]*fac
    m2=make_core_model(p2)
    o2=model_array(m2.flux_density(np.array([2e5,3e5]), NU_I*np.ones(2)).total)*1e26*att
    o1=model_array(m.flux_density(np.array([2e5,3e5]), NU_I*np.ones(2)).total)*1e26*att
    x2=model_array(m2.flux(np.array([1.36e5,4.06e5]), XRT_BAND[0], XRT_BAND[1],10).total)
    x1=model_array(m.flux(np.array([1.36e5,4.06e5]), XRT_BAND[0], XRT_BAND[1],10).total)
    print(f"  E x{fac:4.1f}: dmag_opt(2e5)={-2.5*np.log10(o2[0]/o1[0]):6.3f}  dmag_opt(3e5)={-2.5*np.log10(o2[1]/o1[1]):6.3f}  "
          f"X-ray x{x2[0]/x1[0]:5.2f}/{x2[1]/x1[1]:5.2f}  dE_iso={(fac-1)*BASE['E_iso_core']:.3e} erg")
print("\n=== density-bump sensitivity (nu_opt vs nu_c test) ===")
for fac in [3.,6.,10.]:
    p2=dict(BASE); p2["n_ism"]=BASE["n_ism"]*fac
    m2=make_core_model(p2)
    o2=model_array(m2.flux_density(np.array([2e5]), NU_I*np.ones(1)).total)*1e26*att
    o1=model_array(m.flux_density(np.array([2e5]), NU_I*np.ones(1)).total)*1e26*att
    print(f"  n x{fac:4.1f}: dmag_opt(2e5)={-2.5*np.log10(o2[0]/o1[0]):6.3f}")
