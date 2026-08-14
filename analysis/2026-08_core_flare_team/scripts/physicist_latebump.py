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
from grb.const import REDSHIFT, XRT_NU_LO, XRT_NU_HI

OFF = {"VT/B":2.0, "r":0.0, "R":0.0, "VT/R":0.0, "i":-1.0, "I":-1.0, "z":-2.0, "y":-3.0, "green":1.0}
d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")

BASE = dict(E_iso_core=1.08e52, Gamma0_core=136.0, theta_c_core=0.760, n_ism=134.0,
            p=2.121, eps_e=0.034, eps_B=0.030, xi=0.31155838, tau=46.5,
            p_r=2.7705, eps_e_r=0.1084, eps_B_r=0.528, xi_r=0.88184)
FL = (2087., 98.6, 7447., 4.72e-10, 0.68325)
AV = 0.03585
NU = {"r":4.813e14, "R":4.677e14, "i":3.932e14, "z":3.363e14, "y":3.005e14}

m = make_core_model(BASE)
def model_mag(band, t):
    nu = NU[band]
    f = model_array(m.flux_density(t, nu*np.ones_like(t)).total)*1e26
    b = FL[4]
    fl = norris_flare(t, *FL[:4])
    K = fl*(1-b)/(XRT_NU_HI**(1-b)-XRT_NU_LO**(1-b))
    f = f + K*nu**(-b)*1e26
    f *= host_extinction_attenuation(nu, AV, REDSHIFT)
    return f  # mJy

print("band  log t bin   n   obs_mag(true AB)  model_mag  resid(obs-mod)  obs_mJy  model_mJy  excess_mJy")
edges = 10**np.arange(4.6, 6.35, 0.15)
for band in ["r","R","i","z","y"]:
    if band+"_t" not in d.files: continue
    t = d[band+"_t"]; mg = d[band+"_m"] - OFF[band]     # true AB
    for lo,hi in zip(edges[:-1], edges[1:]):
        k = (t>=lo)&(t<hi)
        if k.sum()<3: continue
        tm = np.median(t[k]); om = np.median(mg[k])
        mm = model_mag(band, np.array([tm]))
        mmj = float(np.atleast_1d(mm)[0])
        mmag = -2.5*np.log10(mmj*1e-3/3631.0)
        omj = 3631.0*10**(-0.4*om)*1e3
        print(f"{band:5s} {np.log10(tm):6.2f} {k.sum():5d}   {om:8.2f}      {mmag:8.2f}   {om-mmag:+8.2f}   "
              f"{omj:9.4f} {mmj:9.4f}  {omj-mmj:+9.4f}")
    print()
