"""Joint 26-D emcee probe of the 'core carries the late XRT' scenario.

Walkers seeded from the incumbent joint optimum (logP=-436.65) AND from the
core-only XRT solutions (hard p~1.7 wide core; slow-cooling p~2.4), hybridised
with the incumbent wing/RS/A_V. Box widened to span all branches.
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
LOGP=[p.scale is Scale.LOG for p in pdefs]

def widen(name, lo, hi):
    for i,p in enumerate(pdefs):
        if p.name==name:
            pdefs[i]=ParamDefWithPrior(name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
WIDE = {
    "E_iso_core":(5e50,1e54), "Gamma0_core":(100,2000), "theta_c_core":(0.001,0.6),
    "n_ism":(0.01,1000), "p":(1.5,3.2), "eps_e":(0.001,0.5), "eps_B":(1e-7,0.15),
    "xi":(0.1,1.0),
    "t_start_flare":(1000,5000), "tau_rise_flare":(5,2000), "tau_decay_flare":(500,20000),
    "A_flare":(1e-10,5e-9), "flare_beta":(0.3,1.5),
    "E_iso_wing":(1e50,1e53), "Gamma0_wing":(5,100), "theta_c_wing":(0.2,1.0),
    "p_wing":(1.8,3.3), "eps_e_wing":(0.01,1.0), "eps_B_wing":(1e-5,0.1),
    "xi_wing":(0.1,1.0),
}
for k,(lo,hi) in WIDE.items(): widen(k,lo,hi)

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

cx=json.load(open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/corexrt.json"))
res={r[0]: r[2] for r in cx["results"]}
def hybrid(tag):
    x=base.copy()
    for l,v in zip(cx["free"], res[tag]):
        x[IX[l]]=v
    return x
HPn, HPw, SCw = hybrid("HP_narrow"), hybrid("HP_wide"), hybrid("SC_wide")

nwalkers, nsteps, nworkers = 64, 700, 16
rng=np.random.default_rng(11)
def cloud(center, n, scat=0.04):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        s=scat if LOGP[i] else max(abs(center[i])*scat,1e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),lo+1e-6,hi-1e-6)
    return pos

pos0=np.vstack([cloud(base,20),cloud(HPn,16),cloud(HPw,12),cloud(SCw,16)])
assert pos0.shape==(nwalkers,len(labels))

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,len(labels),log_probability,
        args=(pdefs,xrt_data,optical_datasets,True,True,xrt_index_data),pool=pool)
    for i,_ in enumerate(sampler.sample(pos0, iterations=nsteps)):
        if (i+1)%50==0:
            lps=sampler.get_log_prob()
            ch=sampler.get_chain()
            cur=lps[-1]
            # basin occupancy by current p value
            pnow=ch[-1][:,IX["p"]]
            print(f"step {i+1}: best {lps.max():.1f} | cur max {cur.max():.1f} | "
                  f"walkers p<2:{np.sum(pnow<2)} 2-2.3:{np.sum((pnow>=2)&(pnow<2.3))} >2.3:{np.sum(pnow>=2.3)}",
                  flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
ib=np.argmax(lp); bx=chain[ib]
print(f"\nBEST logP = {lp[ib]:.2f}  (incumbent -436.65)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/joint_best.npy",bx)
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/joint_chain_tail.npy",sampler.get_chain()[-100:])
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/joint_lp_tail.npy",sampler.get_log_prob()[-100:])
json.dump({"best":float(lp[ib])},open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/jointprobe.json","w"))
