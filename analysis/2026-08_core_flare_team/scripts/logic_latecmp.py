import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction
C=2.99792458e18
FR="/data/dtak/research/grb/GRB251013C/modeling/fit_results"
CAND=[("FLARE-X", np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy"), False),
      ("incumbent", np.load(f"{FR}/final_flare_wing_20260802_131026/top_k_params.npy")[0], True)]
d=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
BANDS={"r":(6215,0.0),"z":(8700,-2.0)}
tg=np.geomspace(5e4,1.2e6,12)
for band,(wl,off) in BANDS.items():
    A=float(galactic_extinction(np.array([wl]))[0])
    t,m=d[band+"_t"],d[band+"_m"]
    print(f"\n=== {band}-band (plotted mag, offset {off:+.0f}; galactic A={A:.3f} added to model) ===")
    print(f"{'t [s]':>10}{'obs':>8}{'n':>5}" + "".join(f"{c[0]:>12}" for c in CAND) + f"{'FX-inc':>9}{'FX-obs':>9}")
    mods={}
    for tag,th,wing in CAND:
        pd=make_param_defs(True,wing)
        pr={p.name:(10**v if p.scale is Scale.LOG else v) for p,v in zip(pd,th)}
        comp=compute_model_components(pr,tg,C/wl,None,True,wing)
        mods[tag]=-2.5*np.log10(comp['total'])+16.4+A+off
    for i,tt in enumerate(tg):
        k=(t>tt*0.85)&(t<tt*1.15)
        o=np.median(m[k]) if k.sum()>=5 else np.nan
        row=f"{tt:>10.2e}{o:>8.2f}{k.sum():>5}"+"".join(f"{mods[c[0]][i]:>12.2f}" for c in CAND)
        row+=f"{mods['FLARE-X'][i]-mods['incumbent'][i]:>9.2f}{mods['FLARE-X'][i]-o:>9.2f}"
        print(row)
