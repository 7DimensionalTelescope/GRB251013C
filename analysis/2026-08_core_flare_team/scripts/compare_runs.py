"""Uniform comparison of the recent fits + external sample.png check.

Candidates (all scored with the committed worktree likelihood, which is
numerically identical to the user's uncommitted one when cal params are absent):
  A final_flare_20260731_172453   core+RS+flare, NO wing (26-vec, 7 dead wing dims)
  B final_flare_wing_20260731_142216  core+RS+flare+wing
  C final_flare_wing_20260730_171914  core+RS+flare+wing (previous)
  D polished INITIAL_GUESS (logP -436.65, PR #4)
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction

pdefs=make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]
xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()

FR="/data/dtak/research/grb/GRB251013C/modeling/fit_results"
def bestvec(run):
    return np.load(f"{FR}/{run}/top_k_params.npy")[0]

GUESS_THETA=np.array([np.log10(v) if p.scale is Scale.LOG else v for p,v in zip(pdefs,[
 6.6226e51,402.145,0.0784868,527.968,2.15362,0.0670652,0.00396854,0.999,20.0651,
 2.995,0.0362407,0.266559,0.999,0.247939,2473.39,101.697,2097.6,1.22857e-9,0.647954,
 1.42319e52,14.8775,0.651733,3.295,0.31909,0.00346853,0.999])])

CAND=[
 ("A no-wing 0731",  bestvec("final_flare_20260731_172453"), False),
 ("B wing 0731",     bestvec("final_flare_wing_20260731_142216"), True),
 ("C wing 0730",     bestvec("final_flare_wing_20260730_171914"), True),
 ("D PR4 guess",     GUESS_THETA, True),
]

print(f"{'model':<15}{'logP':>9}{'XRTchi2':>9}{'optchi2':>9}{'SIchi2':>8}  late XRT resid")
best_params={}
for tag,th,wing in CAND:
    lp=log_probability(th,pdefs,xrt_data,optical_datasets,True,wing,xrt_index_data)
    params={p.name:(10**v if p.scale is Scale.LOG else v) for p,v in zip(pdefs,th)}
    xm,om,si=compute_model_flux_all_bands(params,xrt_data,optical_datasets,True,wing,xrt_index_data)
    xc=np.sum(((xrt_data['flux']-xm)/xrt_data['flux_error'])**2)
    oc=sum(np.sum(((d['flux_mJy']-m)/d['flux_err'])**2) for d,m in zip(optical_datasets,om))
    r=(xrt_data['flux']-xm)/xrt_data['flux_error']
    late=[f"{r[i]:+.1f}" for i in np.where(xrt_data['time']>1e5)[0]]
    print(f"{tag:<15}{lp:>9.1f}{xc:>9.1f}{oc:>9.1f}{si:>8.1f}  {late}")
    best_params[tag]=(params,wing)

# ---- overlay on sample.png parsed data (r and z bands, most telling late) ----
d=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
C_AA=2.99792458e18
tgrid=np.geomspace(1e2,2.5e6,300)
BANDS={"r":(6215,0,"red"),"i":(7545,-1,"gold"),"z":(8700,-2,"gray")}
STYLE={"A no-wing 0731":("-",2.2),"B wing 0731":("--",1.8),"C wing 0730":(":",1.5),"D PR4 guess":("-.",1.5)}
fig,ax=plt.subplots(figsize=(11,7.5))
for band,(wl,off,col) in BANDS.items():
    t,m=d[f"{band}_t"],d[f"{band}_m"]
    ax.scatter(t,m,s=2,alpha=0.2,color=col)
    A=float(galactic_extinction(np.array([wl]))[0])
    for tag,(params,wing) in best_params.items():
        comp=compute_model_components(params,tgrid,C_AA/wl,None,True,wing)
        ls,lw=STYLE[tag]
        ax.plot(tgrid,-2.5*np.log10(comp['total'])+16.4+A+off,ls=ls,lw=lw,color=col,
                label=f"{band}: {tag}" if band=="r" else None)
ax.set_xscale('log'); ax.set_ylim(25,12.5)
ax.set_xlabel('Time since trigger [s]'); ax.set_ylabel('Plotted AB mag (r+0, i-1, z-2)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Recent fits vs sample.png data (r/i/z)')
fig.tight_layout(); fig.savefig("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/runs_vs_sample.png",dpi=150)
print("saved runs_vs_sample.png")
