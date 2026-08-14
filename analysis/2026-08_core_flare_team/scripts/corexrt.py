"""Can the CORE (+flare) alone carry the XRT light curve AND photon index?

Fits core+flare against XRT flux + spectral index ONLY (optical ignored;
wing/RS/A_V frozen at the current INITIAL_GUESS). Multi-start over:
  - slow-cooling seeds (nu_c above XRT: low eps_B/low n, p ~ 2.76 so
    Gamma = (p+1)/2 = 1.88)
  - hard-spectrum seeds (p ~ 1.86 above nu_c: Gamma = p/2+1 = 1.93, wide
    core for a late jet break)
Powell WITHOUT scipy bounds (box enforced by penalty; bounded Powell is
non-monotone on this problem).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data, make_core_model, make_wing_model
from grb.likelihood import log_likelihood, spectral_index_model, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index
from scipy.optimize import minimize
from multiprocessing import Pool

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]

GUESS = {
    "E_iso_core": 6.6226e51, "Gamma0_core": 402.145, "theta_c_core": 0.0784868,
    "n_ism": 527.968, "p": 2.15362, "eps_e": 0.0670652, "eps_B": 0.00396854,
    "xi": 0.999, "tau": 20.0651, "p_r": 2.995, "eps_e_r": 0.0362407,
    "eps_B_r": 0.266559, "xi_r": 0.999, "A_V": 0.247939,
    "t_start_flare": 2473.39, "tau_rise_flare": 101.697, "tau_decay_flare": 2097.6,
    "A_flare": 1.22857e-9, "flare_beta": 0.647954, "E_iso_wing": 1.42319e52,
    "Gamma0_wing": 14.8775, "theta_c_wing": 0.651733, "p_wing": 3.295,
    "eps_e_wing": 0.31909, "eps_B_wing": 0.00346853, "xi_wing": 0.999,
}
base=np.array([np.log10(GUESS[p.name]) if p.scale is Scale.LOG else GUESS[p.name] for p in pdefs])

FREE=["log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism","p",
      "log10_eps_e","log10_eps_B","xi",
      "log10_t_start_flare","log10_tau_rise_flare","log10_tau_decay_flare",
      "log10_A_flare","flare_beta"]
FI=[IX[l] for l in FREE]

SPEC={  # physical bounds for the test box
 "log10_E_iso_core":(1e50,1e55),"log10_Gamma0_core":(100,2000),
 "log10_theta_c_core":(0.01,0.5),"log10_n_ism":(0.01,1000),
 "p":(1.5,3.2),"log10_eps_e":(0.001,0.5),"log10_eps_B":(1e-7,0.1),"xi":(0.1,1.0),
 "log10_t_start_flare":(1000,5000),"log10_tau_rise_flare":(10,2000),
 "log10_tau_decay_flare":(500,20000),"log10_A_flare":(1e-10,5e-9),"flare_beta":(0.3,1.5),
}
BF=np.array([[np.log10(SPEC[l][0]),np.log10(SPEC[l][1])] if l.startswith("log10_")
             else list(SPEC[l]) for l in FREE])

def ll_xrt(th):
    return log_likelihood(th,pdefs,xrt_data,[],True,True,xrt_index_data)

def target(free):
    if np.any(free<BF[:,0]) or np.any(free>BF[:,1]): return 1e12
    th=base.copy(); th[FI]=free
    r=ll_xrt(th)
    return 1e12 if not np.isfinite(r) else -r

def mk(**kw):
    x=base.copy()
    for k,v in kw.items():
        i=IX[k] if k in IX else IX["log10_"+k]
        x[i]=np.log10(v) if LOGP[i] else v
    return x[FI]

seeds=[
 ("control", base[FI].copy()),
 ("SC_lown",  mk(E_iso_core=3e52,Gamma0_core=400,theta_c_core=0.1,n_ism=1.0,p=2.76,eps_e=0.1,eps_B=1e-5,xi=0.9)),
 ("SC_lowB",  mk(E_iso_core=1e53,Gamma0_core=300,theta_c_core=0.05,n_ism=10,p=2.76,eps_e=0.05,eps_B=1e-6,xi=0.9)),
 ("SC_wide",  mk(E_iso_core=3e52,Gamma0_core=400,theta_c_core=0.2,n_ism=100,p=2.9,eps_e=0.1,eps_B=1e-5,xi=0.9)),
 ("HP_wide",  mk(theta_c_core=0.3,n_ism=100,p=1.86)),
 ("HP_wide2", mk(E_iso_core=2e52,Gamma0_core=300,theta_c_core=0.4,n_ism=10,p=1.86,eps_e=0.03,eps_B=0.02,xi=0.9)),
 ("HP_narrow",mk(p=1.86)),
]

def run(job):
    tag,x0=job
    x0=np.clip(np.asarray(x0,float),BF[:,0]+1e-9,BF[:,1]-1e-9)
    v0=-target(x0)
    r=minimize(target,x0,method="Powell",options=dict(maxfev=8000,xtol=1e-4,ftol=1e-7))
    print(f"[done] {tag}: {v0:.1f} -> {-r.fun:.1f} ({r.nfev} evals)",flush=True)
    return tag,float(-r.fun),r.x.tolist()

print(f"{len(seeds)} seeds; control XRT+SI logL = {-target(base[FI]):.2f}",flush=True)
with Pool(len(seeds)) as pool: out=pool.map(run,seeds)
out.sort(key=lambda x:-x[1])
json.dump({"free":FREE,"results":out},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/corexrt.json","w"))

print(f"\n{'seed':<11}{'logL(X+SI)':>11}  details")
for tag,v,x in out:
    th=base.copy(); th[FI]=np.array(x)
    params={p.name:(10**t if p.scale is Scale.LOG else t) for p,t in zip(pdefs,th)}
    xm,_,si=compute_model_flux_all_bands(params,xrt_data,[],True,True,xrt_index_data)
    r=(xrt_data['flux']-xm)/xrt_data['flux_error']
    late=np.where(xrt_data['time']>1e5)[0]
    core=make_core_model(params); wing=make_wing_model(params)
    bm,_=spectral_index_model(core,wing,params,np.array([38.,113.])*3600,True)
    xr=np.sum(r**2)
    print(f"{tag:<11}{v:>11.1f}  XRTchi2={xr:.1f} SIchi2={si:.1f} "
          f"late_resid={[f'{r[i]:+.1f}' for i in late]} lateGamma={(1-bm).round(2)} "
          f"p={params['p']:.2f} eB={params['eps_B']:.2g} n={params['n_ism']:.3g} th={params['theta_c_core']:.3f}")
