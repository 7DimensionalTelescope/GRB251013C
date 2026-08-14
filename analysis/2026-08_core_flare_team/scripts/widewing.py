"""CONTROL EXPERIMENT (logic audit E.1): wing architecture in the SAME wide box
FLARE-X was found in, real fitted data only. 26-param emcee.

Prediction recorded by the auditor: theta_c_core leaves 0.08, p_wing leaves the
3.3 rail, and the wing's late nosedive disappears.
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
WIDE={
 "E_iso_core":(1e51,1e55),"Gamma0_core":(100,2000),"theta_c_core":(0.02,0.8),
 "n_ism":(0.01,3000),"p":(2.01,3.2),"eps_e":(0.001,0.5),"eps_B":(1e-7,0.3),"xi":(0.1,1.0),
 "tau":(5,100),"p_r":(2.0,3.5),"eps_e_r":(0.005,0.5),"eps_B_r":(1e-4,0.6),"xi_r":(0.3,1.0),
 "t_start_flare":(500,2e4),"tau_rise_flare":(5,5000),"tau_decay_flare":(500,1e5),
 "A_flare":(1e-11,2e-8),"flare_beta":(0.0,2.0),
 "E_iso_wing":(1e50,1e53),"Gamma0_wing":(3,100),"theta_c_wing":(0.2,1.0),
 "p_wing":(2.01,3.5),"eps_e_wing":(0.005,1.0),"eps_B_wing":(1e-6,0.1),"xi_wing":(0.1,1.0),
}
for k,(lo,hi) in WIDE.items(): widen(k,lo,hi)

INC=np.load("/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026/top_k_params.npy")[0]
FX=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")  # 19-dim = first 19 of wing layout

def hybrid(wing_phys):
    x=INC.copy()
    x[:19]=FX
    x[IX["p"]]=max(FX[IX["p"]],2.02)
    for k,v in wing_phys.items():
        i=IX["log10_"+k] if "log10_"+k in IX else IX[k]
        x[i]=np.log10(v) if LOGP[i] else v
    return x

# small, late-peaking, X-ray-quiet wing (two-component reading of the bump);
# per design review its wing is negligible -> it doubles as the run's own
# commensurable no-wing baseline (scores -793.3 inside THIS pdefs box).
HYB_A=hybrid(dict(E_iso_wing=2e51,Gamma0_wing=8.0,theta_c_wing=0.5,p_wing=2.5,
                  eps_e_wing=0.1,eps_B_wing=1e-5,xi_wing=0.5))

nwalkers,nsteps,nworkers=64,2400,16
rng=np.random.default_rng(53)
def cloud(center,n,scat=0.05):
    pos=np.tile(center,(n,1))
    for i,p in enumerate(pdefs):
        lo=np.log10(p.lower) if LOGP[i] else p.lower
        hi=np.log10(p.upper) if LOGP[i] else p.upper
        s=scat if LOGP[i] else max(abs(center[i])*scat,2e-3)
        pos[:,i]=np.clip(center[i]+rng.normal(0,s,n),lo+1e-6,hi-1e-6)
    return pos
pos0=np.vstack([cloud(INC,32),cloud(HYB_A,32)])
assert pos0.shape==(nwalkers,26)

for tag,c in (("INC",INC),("HYB_A",HYB_A)):
    print(f"seed {tag}: logP = {log_probability(c,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data):.2f}",
          flush=True)

with Pool(nworkers) as pool:
    sampler=emcee.EnsembleSampler(nwalkers,26,log_probability,
        args=(pdefs,xrt_data,optical_datasets,True,True,xrt_index_data),pool=pool)
    for i,_ in enumerate(sampler.sample(pos0,iterations=nsteps)):
        if (i+1)%100==0:
            ch=sampler.get_chain()[-1]
            print(f"step {i+1}: best {sampler.get_log_prob().max():.1f} | "
                  f"th_c med {10**np.median(ch[:,IX['log10_theta_c_core']]):.3f} | "
                  f"p_wing med {np.median(ch[:,IX['p_wing']]):.2f}",flush=True)

chain=sampler.get_chain(flat=True); lp=sampler.get_log_prob(flat=True)
ib=np.argmax(lp); bx=chain[ib]
print(f"\nBEST logP = {lp[ib]:.2f}  (narrow-box wing -430.2; no-wing wide-box -793.3)")
# run's own no-wing baseline: wing energetically negligible
iEw=IX["log10_E_iso_wing"]
nw=chain[:,iEw]<50.5
if nw.any():
    print(f"no-wing baseline (E_iso_wing<10^50.5): best {lp[nw].max():.2f} ({nw.sum()} samples)")
wa=chain[:,iEw]>51.0
if wa.any():
    print(f"wing-active (E_iso_wing>10^51): best {lp[wa].max():.2f} ({wa.sum()} samples)")
for i,l in enumerate(labels):
    v=bx[i]; print(f"{l:<24}{v:>10.4f}{(10**v if LOGP[i] else v):>14.6g}")
np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/widewing_best.npy",bx)
for label,mask in (("within 6 of best (all)", lp>lp[ib]-6),
                   ("within 6 AND wing-active", (lp>lp[ib]-6)&wa)):
    good=chain[mask]
    print(f"\n{label}: {len(good)} samples")
    if len(good)<10: continue
    for name in ("log10_theta_c_core","p","p_wing","log10_theta_c_wing","log10_Gamma0_wing",
                 "log10_E_iso_wing","log10_eps_B_wing","log10_n_ism"):
        c=good[:,IX[name]]
        print(f"  {name:<24} 16/50/84%: {np.percentile(c,16):.3f} {np.percentile(c,50):.3f} {np.percentile(c,84):.3f}")
