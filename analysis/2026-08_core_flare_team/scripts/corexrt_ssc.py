"""Core(+flare) vs XRT flux + spectral index, with SSC cooling ON for the core.

Same harness as corexrt.py but grb.likelihood.make_core_model is patched to a
production-identical builder with ssc=True (KN-limited Compton cooling hardens
the late spectrum to Gamma ~ 1.75-1.85 in the right corner).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet, Scale
from grb.const import REDSHIFT, D_L, MODEL_RESOLUTIONS
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data, make_wing_model
from grb.likelihood import log_likelihood, spectral_index_model, compute_model_flux_all_bands
import grb.likelihood as L
from grb.spectral_index import load_xrt_spectral_index
from scipy.optimize import minimize
from multiprocessing import Pool

def make_core_model_ssc(params):
    """Production make_core_model with ssc=True on the forward shock."""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(E_iso=params["E_iso_core"], Gamma0=params["Gamma0_core"],
                    theta_c=params["theta_c_core"], spreading=True,
                    duration=params.get("tau", 10.0))
    fwd = Radiation(eps_e=params["eps_e"], eps_B=params["eps_B"], p=params["p"],
                    xi_e=params["xi"], ssc=True, kn=True)
    rvs = None
    if "p_r" in params and "eps_e_r" in params and "eps_B_r" in params:
        rvs = Radiation(eps_e=params["eps_e_r"], eps_B=params["eps_B_r"], p=params["p_r"],
                        xi_e=params.get("xi_r", params["xi"]), ssc=False, kn=False)
    return Model(jet=jet, medium=medium, observer=observer, fwd_rad=fwd, rvs_rad=rvs,
                 resolutions=MODEL_RESOLUTIONS)

L.make_core_model = make_core_model_ssc  # likelihood now uses SSC core

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
SPEC={
 "log10_E_iso_core":(1e51,3e54),"log10_Gamma0_core":(100,2000),
 "log10_theta_c_core":(0.05,0.6),"log10_n_ism":(1,3000),
 "p":(2.0,3.2),"log10_eps_e":(0.005,0.5),"log10_eps_B":(1e-7,0.01),"xi":(0.1,1.0),
 "log10_t_start_flare":(1000,5000),"log10_tau_rise_flare":(5,2000),
 "log10_tau_decay_flare":(500,20000),"log10_A_flare":(1e-10,5e-9),"flare_beta":(0.3,1.5),
}
BF=np.array([[np.log10(SPEC[l][0]),np.log10(SPEC[l][1])] if l.startswith("log10_")
             else list(SPEC[l]) for l in FREE])

def target(free):
    if np.any(free<BF[:,0]) or np.any(free>BF[:,1]): return 1e12
    th=base.copy(); th[FI]=free
    r=log_likelihood(th,pdefs,xrt_data,[],True,True,xrt_index_data)
    return 1e12 if not np.isfinite(r) else -r

def mk(**kw):
    x=base.copy()
    for k,v in kw.items():
        i=IX[k] if k in IX else IX["log10_"+k]
        x[i]=np.log10(v) if LOGP[i] else v
    return x[FI]

seeds=[
 ("ctrl_ssc", base[FI].copy()),
 ("S1", mk(E_iso_core=1e53,Gamma0_core=400,theta_c_core=0.3,n_ism=500,p=2.6,eps_e=0.1,eps_B=1e-4,xi=0.9)),
 ("S2", mk(E_iso_core=2e53,Gamma0_core=400,theta_c_core=0.4,n_ism=300,p=2.6,eps_e=0.1,eps_B=3e-5,xi=0.9)),
 ("S3", mk(E_iso_core=1e53,Gamma0_core=400,theta_c_core=0.2,n_ism=500,p=2.5,eps_e=0.1,eps_B=1e-4,xi=0.9)),
 ("S4", mk(E_iso_core=3e53,Gamma0_core=300,theta_c_core=0.3,n_ism=100,p=2.7,eps_e=0.15,eps_B=1e-5,xi=0.9)),
 ("S5", mk(E_iso_core=1e53,Gamma0_core=400,theta_c_core=0.5,n_ism=500,p=2.6,eps_e=0.1,eps_B=1e-4,xi=0.9)),
]

def run(job):
    tag,x0=job
    x0=np.clip(np.asarray(x0,float),BF[:,0]+1e-9,BF[:,1]-1e-9)
    v0=-target(x0)
    r=minimize(target,x0,method="Powell",options=dict(maxfev=8000,xtol=1e-4,ftol=1e-7))
    print(f"[done] {tag}: {v0:.1f} -> {-r.fun:.1f} ({r.nfev} evals)",flush=True)
    return tag,float(-r.fun),r.x.tolist()

print(f"{len(seeds)} seeds (SSC core); sync-only core best was -19.3",flush=True)
with Pool(len(seeds)) as pool: out=pool.map(run,seeds)
out.sort(key=lambda x:-x[1])
json.dump({"free":FREE,"results":out},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/corexrt_ssc.json","w"))

print(f"\n{'seed':<9}{'logL(X+SI)':>11}  details")
for tag,v,x in out:
    th=base.copy(); th[FI]=np.array(x)
    params={p.name:(10**t if p.scale is Scale.LOG else t) for p,t in zip(pdefs,th)}
    xm,_,si=compute_model_flux_all_bands(params,xrt_data,[],True,True,xrt_index_data)
    r=(xrt_data['flux']-xm)/xrt_data['flux_error']
    late=np.where(xrt_data['time']>1e5)[0]
    core=make_core_model_ssc(params); wing=make_wing_model(params)
    bm,_=spectral_index_model(core,wing,params,np.array([38.,113.])*3600,True)
    # core i-band at 5 hr (should stay below the ~0.5 mJy data)
    i5=np.asarray(core.flux_density(np.array([5*3600.]),np.array([3.93e14])).total)[0]*1e26
    print(f"{tag:<9}{v:>11.1f}  XRTchi2={np.sum(r**2):.1f} SIchi2={si:.1f} "
          f"late={[f'{r[i]:+.1f}' for i in late]} lateG={(1-bm).round(2)} i5={i5*1e3:.0f}uJy "
          f"p={params['p']:.2f} eB={params['eps_B']:.2g} n={params['n_ism']:.3g} th={params['theta_c_core']:.2f} E={params['E_iso_core']:.2g}")
