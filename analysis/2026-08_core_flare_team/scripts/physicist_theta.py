import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
import sys
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np
from scipy.integrate import quad
from grb.modeling import load_all_optical_data
from grb.likelihood import log_likelihood
from grb.functions import norris_flare
from grb.params import make_param_defs
from VegasAfterglow import Scale

xrt, opt = load_all_optical_data()
from grb.spectral_index import load_xrt_spectral_index
si = load_xrt_spectral_index()

# build a param_def list matching the widened box actually used for FLARE-X
defs = make_param_defs(True, False)
theta0 = np.array([52.03341845, 2.1319984, -0.11918524, 2.1262565, 2.12092578, -1.471047,
                   -1.52301058, 0.31155838, 1.66720046, 2.77048111, -0.96510305, -0.27741778,
                   0.88184081, -1.44551679, 3.31945804, 1.9939548, 3.87196744, -9.32615794,
                   0.68324872])
names = [d.name for d in defs]
ith = names.index("theta_c_core")

def LL(vec):
    return log_likelihood(vec, defs, xrt, opt, True, False, si)

base = LL(theta0)
print(f"baseline logL (no prior) = {base:.2f}   [team lead quotes logP=-793.3 incl. A_V prior]")
print("\ntheta_c scan (all else fixed at FLARE-X):")
print(" theta[rad]  theta[deg]   f_b=1-cos   E_true[erg]   logL      dlogL     t_jet[s]")
c=2.99792458e10; mp=1.6726219e-24; z=0.572; E=1.08e52; n=134.
for th in [0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.760,0.90,1.2]:
    v = theta0.copy(); v[ith] = np.log10(th)
    ll = LL(v)
    fb = 1-np.cos(th)
    tj = (1+z)*(3*E/(32*np.pi*n*mp*c**5))**(1/3.)*th**(8/3.)
    print(f"  {th:6.3f}    {np.degrees(th):6.1f}     {fb:8.4f}   {fb*E:10.3e}  {ll:9.2f} {ll-base:+9.2f}   {tj:9.3e}")

print("\n=== flare energetics ===")
ts,tr,td,A,b = 2087.,98.6,7447.,4.72e-10,0.68325
Ef = quad(lambda t: norris_flare(np.array([t]),ts,tr,td,A)[0], ts+1e-3, 2e5, limit=400)[0]
DL=1.059e28
Eiso_flare = 4*np.pi*DL**2*Ef/(1+z)
print(f"  fluence(0.3-10 keV) = {Ef:.3e} erg/cm2  ->  E_iso,flare = {Eiso_flare:.3e} erg  "
      f"= {Eiso_flare/1.08e52*100:.2f}% of E_iso,core")
tpk = ts+np.sqrt(tr*td)
print(f"  t_peak = {tpk:.0f} s ; decay e-fold = {td:.0f} s ; Delta_t/t = {td/tpk:.2f}")

print("\n=== Eichler-Waxman xi degeneracy (xi -> 1) ===")
xi=0.31155838
print(f"  as fitted : E_iso={1.08e52:.3e}  n={134.:.1f}  eps_e={0.034:.4f}  eps_B={0.030:.4f}  xi={xi:.3f}")
print(f"  xi=1 equiv: E_iso={1.08e52/xi:.3e}  n={134./xi:.1f}  eps_e={0.034*xi:.4f}  eps_B={0.030*xi:.4f}  xi=1.000")
print(f"  RS: eps_B_r/eps_B = {0.528/0.030:.1f}  ->  R_B ; sigma ~ eps_B_r/(1-eps_B_r) = {0.528/(1-0.528):.2f}")

print("\n=== SN check at z=0.572 ===")
mu = 5*np.log10(DL/3.0857e24/10)
print(f"  distance modulus mu = {mu:.2f}")
for M,lab in [(-19.3,"SN1998bw-like Ic-BL peak"),(-18.5,"typical Ic-BL"),(-17.5,"faint Ic")]:
    print(f"  {lab:26s} M={M} -> m_obs = {M+mu:.2f} (before K-corr)")
print("  observed rebrightening peak (r, 2.8e5 s) ~ 20.5 AB")
