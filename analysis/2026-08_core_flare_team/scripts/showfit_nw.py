"""Render the fast joint-fit results: best case's 3-panel LC (components) +
chi2 breakdown table for all cases."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np, json
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import compute_model_flux_all_bands, log_probability
from grb.spectral_index import load_xrt_spectral_index
from grb.plotting import compute_model_components
from grb.const import XRT_BAND

J=json.load(open("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/jointfast_nowing.json"))
labels=J["labels"]
pdefs=make_param_defs(True,False)
LOGP=[p.scale is Scale.LOG for p in pdefs]
xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()

print(f"{'case':<6}{'logP':>9}{'XRTchi2':>9}{'optchi2':>9}{'SIchi2':>8}  late XRT resid")
rows=[]
for tag,v,th in J["results"]:
    th=np.array(th)
    params={p.name:(10**t if p.scale is Scale.LOG else t) for p,t in zip(pdefs,th)}
    xm,om,si=compute_model_flux_all_bands(params,xrt_data,optical_datasets,True,True,xrt_index_data)
    xc=np.sum(((xrt_data['flux']-xm)/xrt_data['flux_error'])**2)
    oc=sum(np.sum(((d['flux_mJy']-m)/d['flux_err'])**2) for d,m in zip(optical_datasets,om))
    r=(xrt_data['flux']-xm)/xrt_data['flux_error']
    late=[f"{r[i]:+.1f}" for i in np.where(xrt_data['time']>1e5)[0]]
    print(f"{tag:<6}{v:>9.1f}{xc:>9.1f}{oc:>9.1f}{si:>8.1f}  {late}")
    rows.append((tag,v,params))
print("incumbent (final_flare_wing_20260802_131026): -430.2, XRT 65-ish, late +4.3/+2.7")

tag,v,params=rows[0]
lc_names=('i-band','Leavitt_Rc','Leavitt_Ic')
lc=[d for d in optical_datasets if d['name'] in lc_names]
sdt=[d for d in optical_datasets if d['name'].startswith('7DT_')]
tg=np.geomspace(4e2,2.5e6,300)
comp_x=compute_model_components(params,tg,None,XRT_BAND,True,False)

fig=plt.figure(figsize=(17,5))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,.9])
axX,axO,axS=[fig.add_subplot(gs[i]) for i in range(3)]

axX.errorbar(xrt_data['time'],xrt_data['flux'],yerr=xrt_data['flux_error'],fmt='.',color='k',ms=6,alpha=.75,label='XRT')
axX.plot(tg,comp_x['total'],'k-',lw=2,label='Total')
axX.plot(tg,comp_x['core_fs'],'k:',lw=1.5,label='Core FS (wide, single PL)')
axX.plot(tg,comp_x['core_rs'],'c-.',lw=1.2,label='RS')
axX.plot(tg,comp_x['wing'],'b--',lw=1.5,label='Wing')
axX.plot(tg,comp_x['flare'],'r-',lw=1.2,alpha=.7,label='Flare')
axX.set_xscale('log'); axX.set_yscale('log'); axX.set_ylim(5e-15,3e-9)
axX.set_xlabel('t [s]'); axX.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
axX.legend(fontsize=8); axX.grid(alpha=.3); axX.set_title(f'XRT — case {tag} (logP={v:.1f})',fontweight='bold')

colors={'i-band':'red','Leavitt_Rc':'darkorange','Leavitt_Ic':'darkred'}
offs={'i-band':1.0,'Leavitt_Rc':1.5,'Leavitt_Ic':2.0}
for d in lc:
    c=colors[d['name']]; o=offs[d['name']]
    axO.errorbar(d['time'],d['flux_mJy']*o,yerr=d['flux_err']*o,fmt='.',ms=5,color=c,alpha=.7,
                 label=f"{d['name']}"+(f" (x{o})" if o!=1 else ""))
    cm=compute_model_components(params,tg,d['frequency'],None,True,False)
    axO.plot(tg,cm['total']*o,color=c,lw=2)
    axO.plot(tg,cm['core_fs']*o,color=c,ls=':',lw=1.2,alpha=.7)
    axO.plot(tg,cm['wing']*o,color=c,ls='--',lw=1.2,alpha=.7)
axO.set_xscale('log'); axO.set_yscale('log'); axO.set_ylim(3e-2,50)
axO.set_xlabel('t [s]'); axO.set_ylabel('Flux density [mJy]')
axO.legend(fontsize=8); axO.grid(alpha=.3)
axO.set_title('Optical (dotted=core, dashed=wing)',fontweight='bold')

C_AA=2.99792458e18
wl=np.array([C_AA/d['frequency'] for d in sdt]); fx=np.array([d['flux_mJy'][0] for d in sdt])
fe=np.array([d['flux_err'][0] for d in sdt]); ts=float(np.median([d['time'][0] for d in sdt]))
wgrid=np.geomspace(wl.min()*.9,wl.max()*1.1,120)
sed_t=np.full_like(wgrid,ts)
cm=compute_model_components(params,sed_t,C_AA/wgrid,None,True,False)
axS.errorbar(wl,fx,yerr=fe,fmt='.',color='slategray',ms=6,label='7DT')
axS.plot(wgrid,cm['total'],'k-',lw=2,label=f'Total @{ts/3600:.1f} hr')
axS.plot(wgrid,cm['core_fs'],'k:',lw=1.2,label='Core FS')
axS.plot(wgrid,cm['wing'],'b--',lw=1.2,label='Wing')
axS.set_xscale('log'); axS.set_yscale('log')
axS.set_xlabel(r'Wavelength [$\AA$]'); axS.set_ylabel('Flux density [mJy]')
axS.legend(fontsize=8); axS.grid(alpha=.3); axS.set_title('7DT spectrum',fontweight='bold')

fig.suptitle(f'Fast joint fit — single-power-law wide core + wing + flare (best: {tag})',fontweight='bold')
fig.tight_layout()
fig.savefig("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/nowing_flare_lc.png",dpi=150)
print("saved nowing_flare_lc.png")
