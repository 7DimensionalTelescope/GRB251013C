"""Map the core-FS parameter region that explains early+late XRT as a single
power law (no wing in X-ray).

emcee over core (8) + flare (5) params vs XRT flux + spectral index ONLY.
RS frozen at production values (negligible past 0.13 hr); A_V irrelevant.
Walkers seeded across the three known branches:
  HP  hard p<2, wide jet, high eps_B   (Gamma = p/2+1 ~ 1.85, above nu_c)
  SC  slow cooling, low eps_B          (Gamma = (p+1)/2 ~ 1.8, below nu_c)
  MID p~2.2, mid eps_B                 (Gamma ~ 2.05)
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
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_likelihood
from grb.spectral_index import load_xrt_spectral_index

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
IX={l:i for i,l in enumerate(labels)}
LOGP=[p.scale is Scale.LOG for p in pdefs]

GUESS_TH=np.array([np.log10(v) if p.scale is Scale.LOG else v for p,v in zip(pdefs,[
 6.6226e51,402.145,0.0784868,527.968,2.15362,0.0670652,0.00396854,0.999,20.0651,
 2.995,0.0362407,0.266559,0.999,0.247939,2473.39,101.697,2097.6,1.22857e-9,0.647954,
 1.42319e52,14.8775,0.651733,3.295,0.31909,0.00346853,0.999])])

FREE=["log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism","p",
      "log10_eps_e","log10_eps_B","xi",
      "log10_t_start_flare","log10_tau_rise_flare","log10_tau_decay_flare",
      "log10_A_flare","flare_beta"]
FI=[IX[l] for l in FREE]
SPEC={
 "log10_E_iso_core":(1e51,1e55),"log10_Gamma0_core":(100,2000),
 "log10_theta_c_core":(0.02,0.8),"log10_n_ism":(0.01,3000),
 "p":(1.6,3.2),"log10_eps_e":(0.001,0.5),"log10_eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "log10_t_start_flare":(1000,5000),"log10_tau_rise_flare":(5,2000),
 "log10_tau_decay_flare":(500,2e4),"log10_A_flare":(1e-11,5e-9),"flare_beta":(0.3,1.5),
}
BF=np.array([[np.log10(SPEC[l][0]),np.log10(SPEC[l][1])] if l.startswith("log10_")
             else list(SPEC[l]) for l in FREE])

def logprob(free):
    if np.any(free<BF[:,0]) or np.any(free>BF[:,1]): return -np.inf
    th=GUESS_TH.copy(); th[FI]=free
    return log_likelihood(th,pdefs,xrt_data,[],True,True,xrt_index_data)

cx=json.load(open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/corexrt.json"))
seeds={r[0]:np.array(r[2]) for r in cx["results"]}

nwalkers,nsteps,nworkers=64,800,12
rng=np.random.default_rng(31)
def cloud(center,n,scat=0.05):
    pos=np.tile(center,(n,1))
    for i,l in enumerate(FREE):
        s=scat if l.startswith("log10_") else max(abs(center[i])*scat,2e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),BF[i,0]+1e-6,BF[i,1]-1e-6)
    return pos
pos0=np.vstack([cloud(seeds["HP_narrow"],14),cloud(seeds["HP_wide"],10),
                cloud(seeds["SC_wide"],14),cloud(seeds["SC_lowB"],10),
                cloud(seeds["control"],16)])
assert pos0.shape==(nwalkers,13)

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,13,logprob,pool=pool)
    for i,_ in enumerate(sampler.sample(pos0,iterations=nsteps)):
        if (i+1)%100==0:
            print(f"step {i+1}: best {sampler.get_log_prob().max():.1f}",flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
keep=lp>lp.max()-8   # ~ within joint 8 logL of best: the viable region
good=chain[keep]; glp=lp[keep]
print(f"\nbest logL(XRT+SI) = {lp.max():.2f}; {keep.sum()} samples within 8")

FI_NAME={l:i for i,l in enumerate(FREE)}
eB=good[:,FI_NAME["log10_eps_B"]]
pv=good[:,FI_NAME["p"]]
# branch split: SC = low eps_B (nu_c above band); HP = hard p; MID = rest
bSC = eB < -3.5
bHP = (~bSC)&(pv<2.0)
bMID= (~bSC)&(~bHP)
def rng_str(x,logscale):
    lo,med,hi=np.percentile(x,[5,50,95])
    if logscale: return f"{10**lo:.3g} .. {10**hi:.3g} (med {10**med:.3g})"
    return f"{lo:.2f} .. {hi:.2f} (med {med:.2f})"
for bname,mask in (("SC (low eps_B, slow cooling)",bSC),("HP (hard p<2)",bHP),("MID (p>2, mid eps_B)",bMID)):
    if mask.sum()<20:
        print(f"\n{bname}: only {mask.sum()} samples"); continue
    print(f"\n{bname}: {mask.sum()} samples, best {glp[mask].max():.1f}")
    for l in ("log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism",
              "p","log10_eps_e","log10_eps_B","xi"):
        print(f"  {l.replace('log10_',''):<14}{rng_str(good[mask][:,FI_NAME[l]],l.startswith('log10_'))}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_good.npy",good)
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_lp.npy",glp)
