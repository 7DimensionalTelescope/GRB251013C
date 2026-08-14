"""FLARE-X (no-wing best) figures:
1. corrected 3-panel LC with the flare drawn as its own component (no wing).
2. overlay on parsed sample.png r/i/z data, incumbent shown for reference.
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
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction
from grb.const import XRT_BAND

pd_nw=make_param_defs(True,False)
bx=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
P={p.name:(10**v if p.scale is Scale.LOG else v) for p,v in zip(pd_nw,bx)}

pd_w=make_param_defs(True,True)
INC=np.load("/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026/top_k_params.npy")[0]
PI={p.name:(10**v if p.scale is Scale.LOG else v) for p,v in zip(pd_w,INC)}

xrt_data, optical_datasets = load_all_optical_data()
tg=np.geomspace(4e2,2.5e6,300)

# ---------- figure 1: corrected components ----------
cx=compute_model_components(P,tg,None,XRT_BAND,True,False)
fig=plt.figure(figsize=(17,5))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,.9])
axX,axO,axS=[fig.add_subplot(gs[i]) for i in range(3)]
axX.errorbar(xrt_data['time'],xrt_data['flux'],yerr=xrt_data['flux_error'],fmt='.',color='k',ms=6,alpha=.75,label='XRT')
axX.plot(tg,cx['total'],'k-',lw=2,label='Total')
axX.plot(tg,cx['core_fs'],'k:',lw=1.5,label='Core FS (wide, single PL)')
axX.plot(tg,cx['core_rs'],'c-.',lw=1.2,label='RS')
axX.plot(tg,cx['flare'],'r--',lw=1.5,label='Flare')
axX.set_xscale('log'); axX.set_yscale('log'); axX.set_ylim(5e-15,3e-9)
axX.set_xlabel('t [s]'); axX.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
axX.legend(fontsize=8); axX.grid(alpha=.3)
axX.set_title('XRT — no-wing FLARE-X (logP=-793.3)',fontweight='bold')

colors={'i-band':'red','Leavitt_Rc':'darkorange','Leavitt_Ic':'darkred'}
offs={'i-band':1.0,'Leavitt_Rc':1.5,'Leavitt_Ic':2.0}
for d in [x for x in optical_datasets if x['name'] in colors]:
    c=colors[d['name']]; o=offs[d['name']]
    axO.errorbar(d['time'],d['flux_mJy']*o,yerr=d['flux_err']*o,fmt='.',ms=5,color=c,alpha=.7,
                 label=f"{d['name']}"+(f" (x{o})" if o!=1 else ""))
    cm=compute_model_components(P,tg,d['frequency'],None,True,False)
    axO.plot(tg,cm['total']*o,color=c,lw=2)
    axO.plot(tg,cm['core_fs']*o,color=c,ls=':',lw=1.2,alpha=.8)
    axO.plot(tg,cm['flare']*o,color=c,ls='--',lw=1.2,alpha=.8)
axO.set_xscale('log'); axO.set_yscale('log'); axO.set_ylim(3e-2,50)
axO.set_xlabel('t [s]'); axO.set_ylabel('Flux density [mJy]')
axO.legend(fontsize=8); axO.grid(alpha=.3)
axO.set_title('Optical (dotted=core, dashed=FLARE; no wing)',fontweight='bold')

sdt=[d for d in optical_datasets if d['name'].startswith('7DT_')]
C_AA=2.99792458e18
wl=np.array([C_AA/d['frequency'] for d in sdt]); fx=np.array([d['flux_mJy'][0] for d in sdt])
fe=np.array([d['flux_err'][0] for d in sdt]); ts=float(np.median([d['time'][0] for d in sdt]))
wgrid=np.geomspace(wl.min()*.9,wl.max()*1.1,120)
cm=compute_model_components(P,np.full_like(wgrid,ts),C_AA/wgrid,None,True,False)
axS.errorbar(wl,fx,yerr=fe,fmt='.',color='slategray',ms=6,label='7DT')
axS.plot(wgrid,cm['total'],'k-',lw=2,label=f'Total @{ts/3600:.1f} hr')
axS.plot(wgrid,cm['core_fs'],'k:',lw=1.2,label='Core FS')
axS.plot(wgrid,cm['flare'],'r--',lw=1.2,label='Flare')
axS.set_xscale('log'); axS.set_yscale('log')
axS.set_xlabel(r'Wavelength [$\AA$]'); axS.set_ylabel('Flux density [mJy]')
axS.legend(fontsize=8); axS.grid(alpha=.3); axS.set_title('7DT spectrum',fontweight='bold')
fig.suptitle('No-wing FLARE-X: wide single-PL core + RS + flare (flare shown dashed)',fontweight='bold')
fig.tight_layout(); fig.savefig("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/flarex_lc.png",dpi=150)
print("saved flarex_lc.png")

# ---------- figure 2: vs sample.png ----------
S=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
BANDS={"r":(6215,0,"red"),"i":(7545,-1,"gold"),"z":(8700,-2,"gray")}
fig,ax=plt.subplots(figsize=(11,7.5))
for band,(wlA,off,col) in BANDS.items():
    t,m=S[f"{band}_t"],S[f"{band}_m"]
    ax.scatter(t,m,s=2,alpha=0.2,color=col)
    A=float(galactic_extinction(np.array([float(wlA)]))[0])
    cm=compute_model_components(P,tg,C_AA/wlA,None,True,False)
    ax.plot(tg,-2.5*np.log10(cm['total'])+16.4+A+off,'-',lw=2.2,color=col,
            label=f"{band}: FLARE-X (no wing)" if band=="r" else None)
    ci=compute_model_components(PI,tg,C_AA/wlA,None,True,True)
    ax.plot(tg,-2.5*np.log10(ci['total'])+16.4+A+off,'-.',lw=1.4,color=col,alpha=.8,
            label=f"{band}: incumbent (with wing)" if band=="r" else None)
ax.set_xscale('log'); ax.set_ylim(25,12.5)
ax.set_xlabel('Time since trigger [s]'); ax.set_ylabel('Plotted AB mag (r+0, i-1, z-2)')
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax.set_title('FLARE-X (solid) vs incumbent (dash-dot) vs sample.png data')
fig.tight_layout(); fig.savefig("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/flarex_vs_sample.png",dpi=150)
print("saved flarex_vs_sample.png")
