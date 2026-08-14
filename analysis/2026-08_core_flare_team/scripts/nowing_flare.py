"""No-wing exploration, physical branches only (p >= 2.01), flare box widened.

19-param emcee: core+RS+flare+A_V vs ALL data, include_wing=False.
Flare freedom extended: t_start 500-2e4 s, tau_rise 5-5000, tau_decay 500-1e5,
A up to 2e-8, beta 0-2 - so the flare can try to play the wing's optical role.
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

def widen(name, lo, hi):
    for i,p in enumerate(pdefs):
        if p.name==name:
            pdefs[i]=ParamDefWithPrior(name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
for k,(lo,hi) in {
 "E_iso_core":(1e51,1e55),"Gamma0_core":(100,2000),"theta_c_core":(0.02,0.8),
 "n_ism":(0.01,3000),"p":(2.01,3.2),"eps_e":(0.001,0.5),"eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "tau":(5,100),"p_r":(2.0,3.5),"eps_e_r":(0.005,0.5),"eps_B_r":(1e-4,0.6),"xi_r":(0.3,1.0),
 "t_start_flare":(500,2e4),"tau_rise_flare":(5,5000),"tau_decay_flare":(500,1e5),
 "A_flare":(1e-11,2e-8),"flare_beta":(0.0,2.0),
}.items(): widen(k,lo,hi)

J=json.load(open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/jointfast_nowing.json"))
res={r[0]:np.array(r[2]) for r in J["results"]}
MIDv, SCv = res["MID"], res["SC"]
MIDv=MIDv.copy(); MIDv[IX["p"]]=max(MIDv[IX["p"]],2.02)
SCv=SCv.copy();  SCv[IX["p"]]=max(SCv[IX["p"]],2.02)

def variant(base,**kw):
    x=base.copy()
    for k,v in kw.items():
        i=IX[k] if k in IX else IX["log10_"+k]
        x[i]=np.log10(v) if LOGP[i] else v
    return x

nwalkers,nsteps,nworkers=64,800,12
rng=np.random.default_rng(41)
def cloud(center,n,scat=0.05):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        s=scat if LOGP[i] else max(abs(center[i])*scat,2e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),lo+1e-6,hi-1e-6)
    return pos

pos0=np.vstack([
    cloud(MIDv,18),
    cloud(variant(MIDv,t_start_flare=8000,tau_decay_flare=3e4,A_flare=2e-9),12),  # late flare
    cloud(variant(MIDv,tau_decay_flare=6e4,flare_beta=1.6),10),                    # long soft flare
    cloud(SCv,12),
    cloud(variant(SCv,t_start_flare=6000,tau_decay_flare=4e4),12),
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
print(f"\nBEST logP = {lp[ib]:.2f}  (fast no-wing MID was -1272; incumbent wing -430.2)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy",bx)
good=chain[lp>lp[ib]-6]
print(f"\n{len(good)} samples within 6:")
for name in ("p","log10_theta_c_core","log10_eps_B","log10_t_start_flare",
             "log10_tau_decay_flare","flare_beta","log10_A_flare"):
    c=good[:,IX[name]]
    print(f"  {name:<24} 16/50/84%: {np.percentile(c,16):.3f} {np.percentile(c,50):.3f} {np.percentile(c,84):.3f}")
