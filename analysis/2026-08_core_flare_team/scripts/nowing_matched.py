"""Matched no-wing control: 19-dim core+RS+flare in the IDENTICAL shared box
as widewing.py (its 19 non-wing dims), same walkers (64), same steps (2400),
same data version. Closes the last audit hold: no-wing never had a dedicated
run in this box (widewing's in-run baseline was 17 accidental samples at -740).
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
pdefs = make_param_defs(True, False)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]
assert len(labels)==19

def widen(name, lo, hi):
    for i,p in enumerate(pdefs):
        if p.name==name:
            pdefs[i]=ParamDefWithPrior(name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
# byte-identical to widewing.py WIDE for the 19 shared dims
for k,(lo,hi) in {
 "E_iso_core":(1e51,1e55),"Gamma0_core":(100,2000),"theta_c_core":(0.02,0.8),
 "n_ism":(0.01,3000),"p":(2.01,3.2),"eps_e":(0.001,0.5),"eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "tau":(5,100),"p_r":(2.0,3.5),"eps_e_r":(0.005,0.5),"eps_B_r":(1e-4,0.6),"xi_r":(0.3,1.0),
 "t_start_flare":(500,2e4),"tau_rise_flare":(5,5000),"tau_decay_flare":(500,1e5),
 "A_flare":(1e-11,2e-8),"flare_beta":(0.0,2.0),
}.items(): widen(k,lo,hi)

FX=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
NP766=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_best.npy")     # -766.7 (narrow-core no-wing)
WW=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/widewing_best.npy")[:19]  # -419 solution, wing stripped

def clip_box(x):
    y=x.copy()
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        y[i]=np.clip(y[i],lo+1e-6,hi-1e-6)
    return y
SEEDS=[("FLARE-X",clip_box(FX)),("NP766",clip_box(NP766)),("WW-core",clip_box(WW))]
for tag,c in SEEDS:
    print(f"seed {tag}: logP = {log_probability(c,pdefs,xrt_data,optical_datasets,True,False,xrt_index_data):.2f}",flush=True)

nwalkers,nsteps,nworkers=64,2400,16
rng=np.random.default_rng(61)
def cloud(center,n,scat=0.05):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        s=scat if LOGP[i] else max(abs(center[i])*scat,2e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),lo+1e-6,hi-1e-6)
    return pos
pos0=np.vstack([cloud(SEEDS[0][1],24),cloud(SEEDS[1][1],24),cloud(SEEDS[2][1],16)])
assert pos0.shape==(nwalkers,19)

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,19,log_probability,
        args=(pdefs,xrt_data,optical_datasets,True,False,xrt_index_data),pool=pool)
    for i,_ in enumerate(sampler.sample(pos0,iterations=nsteps)):
        if (i+1)%200==0:
            print(f"step {i+1}: best {sampler.get_log_prob().max():.1f}",flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
ib=np.argmax(lp); bx=chain[ib]
print(f"\nBEST no-wing (matched box/steps) logP = {lp[ib]:.2f}")
print("(wing matched run best: -418.99; accidental in-run baseline was -740.1; FLARE-X -793.3)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_matched_best.npy",bx)
good=chain[lp>lp[ib]-6]
print(f"\n{len(good)} samples within 6:")
for name in ("log10_theta_c_core","p","log10_n_ism","log10_eps_B","log10_tau_decay_flare","flare_beta"):
    c=good[:,IX[name]]
    print(f"  {name:<24} 16/50/84%: {np.percentile(c,16):.3f} {np.percentile(c,50):.3f} {np.percentile(c,84):.3f}")
