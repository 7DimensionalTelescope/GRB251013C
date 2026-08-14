import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import sys
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np
from grb.modeling import make_core_model, load_all_optical_data
from grb.utils import model_array
from grb.extinction import host_extinction_attenuation
from grb.const import REDSHIFT, XRT_BAND

BASE = dict(E_iso_core=1.08e52, Gamma0_core=136.0, theta_c_core=0.760, n_ism=134.0,
            p=2.121, eps_e=0.034, eps_B=0.030, xi=0.31155838, tau=46.5,
            p_r=2.7705, eps_e_r=0.1084, eps_B_r=0.528, xi_r=0.88184)
AV = 0.03585
NU_I = 3.932e14

xrt, opt = load_all_optical_data()
ti = opt[0]["time"]; fi = opt[0]["flux_mJy"]; ei = opt[0]["flux_err"]
mearly = ti < 2087
tgrid = np.logspace(np.log10(60), np.log10(1e4), 60)

def curves(p, tag, tg=tgrid):
    m = make_core_model(p)
    r = m.flux_density(tg, NU_I*np.ones_like(tg))
    att = host_extinction_attenuation(NU_I, AV, REDSHIFT)
    tot = model_array(r.total)*1e26*att
    # forward-only model
    p2 = {k:v for k,v in p.items() if k not in ("p_r","eps_e_r","eps_B_r","xi_r")}
    m2 = make_core_model(p2)
    fs = model_array(m2.flux_density(tg, NU_I*np.ones_like(tg)).total)*1e26*att
    rs = tot - fs
    def alp(y, lo, hi):
        k = (tg>=lo)&(tg<=hi)&(y>0)
        if k.sum()<3: return np.nan
        return -np.polyfit(np.log10(tg[k]), np.log10(y[k]),1)[0]
    print(f"--- {tag}")
    print(f"    i-band tot: t=100 {np.interp(100,tg,tot):8.3f}  300 {np.interp(300,tg,tot):8.3f}  "
          f"1000 {np.interp(1000,tg,tot):8.3f}  2000 {np.interp(2000,tg,tot):8.3f} mJy")
    print(f"    RS only   : t=100 {np.interp(100,tg,rs):8.3f}  300 {np.interp(300,tg,rs):8.3f}  "
          f"1000 {np.interp(1000,tg,rs):8.3f}  2000 {np.interp(2000,tg,rs):8.3f} mJy")
    print(f"    alpha_tot(100-2000)={alp(tot,100,2000):.3f}  alpha_RS(100-2000)={alp(np.maximum(rs,1e-30),100,2000):.3f}  "
          f"alpha_FS={alp(fs,100,2000):.3f}   RSfrac@200s={np.interp(200,tg,rs)/max(np.interp(200,tg,tot),1e-30):.2f}")
    chi2 = np.sum(((fi[mearly]-np.interp(ti[mearly],tg,tot))/ei[mearly])**2)
    print(f"    chi2 vs early i-band (n={mearly.sum()}) = {chi2:.1f}")
    return tot, rs, fs

print("DATA: i-band 94s=%.2f 300s~%.2f 1000s~%.2f 2000s~%.2f mJy ; alpha(94-2087)=0.573"%(
    fi[0], np.interp(300,ti,fi), np.interp(1000,ti,fi), np.interp(2000,ti,fi)))
print()

cfgs = [
    ("A  FLARE-X as-is (tau=46.5)", dict()),
    ("B  tau=300 s", dict(tau=300.)),
    ("C  tau=2000 s", dict(tau=2000.)),
    ("D  tau=2000, Gamma0=60", dict(tau=2000., Gamma0_core=60.)),
    ("E  tau=2000, Gamma0=60, eps_B_r=0.1, p_r=2.2", dict(tau=2000., Gamma0_core=60., eps_B_r=0.1, p_r=2.2)),
    ("F  tau=46.5, eps_B_r=0.3, Gamma0=400", dict(Gamma0_core=400., eps_B_r=0.3)),
    ("G  low-density: n=3, eps_B=0.005, tau=46.5", dict(n_ism=3., eps_B=0.005)),
    ("H  low-density n=3 eps_B=0.005 p=2.0 eps_e=0.1 xi=1", dict(n_ism=3., eps_B=0.005, p=2.0, eps_e=0.1, xi=1.0)),
]
for tag, over in cfgs:
    p = dict(BASE); p.update(over)
    try:
        curves(p, tag)
    except Exception as e:
        print(tag, "FAILED", e)
    print()
