"""No-wing (core+RS+flare, TRUE 19-param model) emcee probe in a broadened box.

Rails hit by final_flare_20260731_172453 released: theta_c_core -> 0.5,
p -> [1.6, 3.0], n_ism -> 1000, tau_decay_flare -> 1e5 (long flare may mimic
the late component), eps_B broad. Seeded from that run's best (first 19 dims)
plus long-flare / wide-hard-core variants.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)

import emcee
from multiprocessing import Pool
from VegasAfterglow import Scale
from grb.params import make_param_defs, ParamDefWithPrior
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability
from grb.spectral_index import load_xrt_spectral_index

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()

pdefs = make_param_defs(include_flare=True, include_wing=False)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
assert len(labels)==19, labels
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]

def widen(name, lo, hi):
    for i,p in enumerate(pdefs):
        if p.name==name:
            pdefs[i]=ParamDefWithPrior(name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
for k,(lo,hi) in {
    "E_iso_core":(1e51,1e54), "Gamma0_core":(100,2000), "theta_c_core":(0.001,0.5),
    "n_ism":(0.1,1000), "p":(1.6,3.0), "eps_e":(0.005,0.5), "eps_B":(1e-6,0.15),
    "xi":(0.1,1.0), "tau":(5,100),
    "p_r":(2.0,3.5), "eps_e_r":(0.005,0.5), "eps_B_r":(1e-4,0.6), "xi_r":(0.3,1.0),
    "t_start_flare":(1000,5000), "tau_rise_flare":(5,2000),
    "tau_decay_flare":(500,1e5), "A_flare":(1e-11,5e-9), "flare_beta":(0.3,1.5),
}.items(): widen(k,lo,hi)

A=np.load("/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_20260731_172453/top_k_params.npy")[0][:19]

nwalkers,nsteps,nworkers=64,800,12
rng=np.random.default_rng(23)
def cloud(center,n,scat=0.04):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        s=scat if LOGP[i] else max(abs(center[i])*scat,1e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),lo+1e-6,hi-1e-6)
    return pos
def variant(**kw):
    x=A.copy()
    for k,v in kw.items():
        i=IX[k] if k in IX else IX["log10_"+k]
        x[i]=np.log10(v) if LOGP[i] else v
    return x

pos0=np.vstack([
    cloud(A,24),
    cloud(variant(tau_decay_flare=5e4,A_flare=2e-10),16),      # long flare
    cloud(variant(theta_c_core=0.3,n_ism=200,p=1.8,eps_B=0.09,eps_e=0.12),12),  # wide hard core
    cloud(variant(theta_c_core=0.2,tau_decay_flare=3e4),12),
])
assert pos0.shape==(nwalkers,19)

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,19,log_probability,
        args=(pdefs,xrt_data,optical_datasets,True,False,xrt_index_data),pool=pool)
    for i,_ in enumerate(sampler.sample(pos0,iterations=nsteps)):
        if (i+1)%100==0:
            print(f"step {i+1}: best {sampler.get_log_prob().max():.1f}",flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
ib=np.argmax(lp); bx=chain[ib]
print(f"\nBEST logP = {lp[ib]:.2f}  (railed no-wing run was -790.8; wing best -436.7)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_best.npy",bx)
good=chain[lp>lp[ib]-5]
print(f"\n{len(good)} samples within 5:")
for name in ("p","log10_theta_c_core","log10_n_ism","log10_eps_B","log10_tau_decay_flare"):
    c=good[:,IX[name]]
    print(f"  {name:<22} 16/50/84%: {np.percentile(c,16):.3f} {np.percentile(c,50):.3f} {np.percentile(c,84):.3f}")
