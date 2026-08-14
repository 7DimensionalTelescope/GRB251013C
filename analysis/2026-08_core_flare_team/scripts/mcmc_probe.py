"""Short emcee probe: does opening p<2 (core) and p_wing<2.2 improve the fit?

Widened box: p in [1.5, 2.3], p_wing in [1.6, 3.3], n_ism up to 1000,
theta_c_wing up to 1.0 (rails released). Walkers seeded mostly around the
final_flare_wing_20260730_171914 best fit, with a minority at low-p /
low-p_wing variants so the sampler can compare basins directly.
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

pdefs = make_param_defs(True, True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}

# widen: rebuild the changed defs in place
def widen(defs, name, lo, hi):
    for i,p in enumerate(defs):
        if p.name==name:
            defs[i]=ParamDefWithPrior(name, lo, hi, p.scale,
                                      gaussian_prior=p.gaussian_prior)
widen(pdefs,"p",1.5,2.3)
widen(pdefs,"p_wing",1.6,3.3)
widen(pdefs,"n_ism",5,1000)
widen(pdefs,"theta_c_wing",0.2,1.0)

RUN="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260730_171914"
best=np.load(RUN+"/top_k_params.npy")[0]

nwalkers, nsteps, nworkers = 64, 600, 16
rng=np.random.default_rng(7)
LOGP=[p.scale is Scale.LOG for p in pdefs]

def cloud(center, n, scat=0.05):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        if LOGP[i]:
            pos[:,i]+=rng.normal(0,scat,n)
        else:
            pos[:,i]+=rng.normal(0,abs(center[i])*scat,n)
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        pos[:,i]=np.clip(pos[:,i],lo+1e-6,hi-1e-6)
    return pos

def variant(**kw):
    x=best.copy()
    for k,v in kw.items(): x[IX[k]]=v
    return x

pos0=np.vstack([
    cloud(best, 28),
    cloud(variant(p=1.85), 12),
    cloud(variant(p_wing=2.3), 8),
    cloud(variant(p=1.85, p_wing=2.3), 8),
    cloud(variant(p=1.85, p_wing=1.9), 8),
])
assert pos0.shape==(nwalkers,len(labels))

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,len(labels),log_probability,
        args=(pdefs,xrt_data,optical_datasets,True,True,xrt_index_data),pool=pool)
    for i, _ in enumerate(sampler.sample(pos0, iterations=nsteps)):
        if (i+1)%50==0:
            lp=sampler.get_log_prob()[-1 if i else 0]
            ch=sampler.get_chain()
            print(f"step {i+1}: best so far {sampler.get_log_prob().max():.2f}, "
                  f"cur max {lp.max():.2f}", flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
ib=np.argmax(lp); bx=chain[ib]
print(f"\nBEST logP = {lp[ib]:.2f}  (run best was -474.78)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.5g}")

# where did p / p_wing go in the good tail?
good=chain[lp>lp[ib]-5]
print(f"\n{len(good)} samples within 5 of best:")
for name in ("p","p_wing","log10_n_ism","log10_theta_c_wing","log10_eps_B"):
    c=good[:,IX[name]]
    print(f"  {name:<20} 16/50/84%: {np.percentile(c,16):.3f} {np.percentile(c,50):.3f} {np.percentile(c,84):.3f}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/probe_best.npy", bx)
json.dump({"best_logp":float(lp[ib])},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/probe.json","w"))
