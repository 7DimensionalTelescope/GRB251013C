"""Three single-power-law core-FS cases (HP / SC / MID) vs XRT + optical data.

Figure: (a) XRT band LC, (b) XRT photon index, (c) i-band optical LC.
Consistency check: core optical flux as a fraction of every observed optical
point (must stay < 1; wing+flare supply the remainder).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grb.modeling import make_core_model, load_all_optical_data
from grb.likelihood import spectral_index_model
from grb.functions import norris_flare
from grb.extinction import host_extinction_attenuation, galactic_extinction
from grb.spectral_index import load_xrt_spectral_index
from grb.utils import model_array
from grb.const import XRT_BAND, REDSHIFT

good=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_good.npy")
lp=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/coremap_lp.npy")
FREE=["log10_E_iso_core","log10_Gamma0_core","log10_theta_c_core","log10_n_ism","p",
      "log10_eps_e","log10_eps_B","xi",
      "log10_t_start_flare","log10_tau_rise_flare","log10_tau_decay_flare",
      "log10_A_flare","flare_beta"]
IX={l:i for i,l in enumerate(FREE)}
eB=good[:,IX["log10_eps_B"]]; pv=good[:,IX["p"]]
bSC=eB<-3.5; bHP=(~bSC)&(pv<2.0); bMID=(~bSC)&(~bHP)

def to_params(x):
    P={}
    for l,v in zip(FREE,x):
        n=l.replace("log10_","")
        P[n]=10**v if l.startswith("log10_") else v
    P["tau"]=20.0
    return P

CASES={}
for name,mask in (("HP",bHP),("SC",bSC),("MID",bMID)):
    CASES[name]=to_params(good[mask][np.argmax(lp[mask])])

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
AV_HOST=0.247939
NU_I=3.93e14

tg=np.geomspace(4e2,2.5e6,300)
COL={"HP":"#c1121f","SC":"#1f6f8b","MID":"#6a994e"}
LAB={"HP":"HP: p=1.7, eps_B=0.07 (hard, above nu_c)",
     "SC":"SC: p=2.4, eps_B=2e-5 (slow cooling)",
     "MID":"MID: p=2.1, eps_B=6e-4"}

fig,axes=plt.subplots(1,3,figsize=(17,5))
axX,axG,axO=axes

# (a) XRT
axX.errorbar(xrt_data['time'],xrt_data['flux'],yerr=xrt_data['flux_error'],
             fmt='.',color='k',ms=6,alpha=0.75,label='XRT data',zorder=5)
# (b) photon index data
gam=1-xrt_index_data['beta']
axG.errorbar(xrt_index_data['time'],gam,
             yerr=[xrt_index_data['beta_err_low'],xrt_index_data['beta_err_high']],
             fmt='o',color='k',ms=4,alpha=0.7,label='XRT photon index',zorder=5)
# (c) optical data: fitted i-band + Leavitt_Ic (+parsed sample.png i, late)
for d,mk,cc,lab in ((next(x for x in optical_datasets if x['name']=='i-band'),'.','#555','i-band (fit data)'),
                    (next(x for x in optical_datasets if x['name']=='Leavitt_Ic'),'s','#999','Leavitt Ic (fit data)')):
    axO.errorbar(d['time'],d['flux_mJy'],yerr=d['flux_err'],fmt=mk,ms=4,color=cc,alpha=0.7,label=lab,zorder=5)
S=np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
ti,mi=S["i_t"],S["i_m"]
sel=ti>3e4
A_gal_i=float(galactic_extinction(np.array([7545.0]))[0])
fi=10**(-((mi[sel]+1)-A_gal_i-16.4)/2.5)   # plotted i-1 -> i, de-extincted, mJy
axO.plot(ti[sel],fi,'.',ms=3,color='goldenrod',alpha=0.4,label='sample.png i (parsed)',zorder=4)

print("Optical consistency: max core/(observed flux) per dataset")
print(f"{'case':<5}"+ "".join(f"{n:>12}" for n in ("i-band","Leav_Rc","Leav_Ic","7DT(med)","parsed_i")))
att_i=host_extinction_attenuation(NU_I,AV_HOST,REDSHIFT)
for name,P in CASES.items():
    m=make_core_model(P)
    fx=model_array(m.flux(tg,XRT_BAND[0],XRT_BAND[1],10).total).copy()
    fx+=norris_flare(tg,P["t_start_flare"],P["tau_rise_flare"],P["tau_decay_flare"],P["A_flare"])
    axX.plot(tg,fx,color=COL[name],lw=2,label=LAB[name])
    bm,_=spectral_index_model(m,None,P,tg,False)
    axG.plot(tg,1-bm,color=COL[name],lw=2)
    fo=model_array(m.flux_density(tg,NU_I*np.ones_like(tg)).total)*1e26*att_i
    axO.plot(tg,fo,color=COL[name],lw=2)
    # fractions
    row=f"{name:<5}"
    for dn in ("i-band","Leavitt_Rc","Leavitt_Ic"):
        d=next(x for x in optical_datasets if x['name']==dn)
        att=host_extinction_attenuation(d['frequency'],AV_HOST,REDSHIFT)
        fm=model_array(m.flux_density(d['time'],d['frequency']*np.ones_like(d['time'])).total)*1e26*att
        row+=f"{np.max(fm/d['flux_mJy']):>12.2f}"
    fr7=[]
    for d in optical_datasets:
        if not d['name'].startswith('7DT_'): continue
        att=host_extinction_attenuation(d['frequency'],AV_HOST,REDSHIFT)
        fm=np.atleast_1d(model_array(m.flux_density(d['time'],d['frequency']*np.ones_like(d['time'])).total))*1e26*att
        fr7.append(float(fm[0]/d['flux_mJy'][0]))
    row+=f"{np.median(fr7):>12.2f}"
    order=np.argsort(ti[sel])
    fmp=model_array(m.flux_density(ti[sel][order],NU_I*np.ones_like(ti[sel][order])).total)*1e26*att_i
    row+=f"{np.max(fmp/fi[order]):>12.2f}"
    print(row)

for ax,t in ((axX,'XRT 0.3-10 keV'),(axG,'XRT photon index'),(axO,'Optical i-band')):
    ax.set_xscale('log'); ax.grid(alpha=0.3); ax.set_xlabel('Time since trigger [s]')
    ax.set_title(t,fontweight='bold')
axX.set_yscale('log'); axX.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
axX.set_ylim(5e-15,3e-9); axX.legend(fontsize=8)
axG.axhspan(1.82-0.21,1.82+0.21,color='k',alpha=0.06)
axG.set_ylim(1.4,2.6); axG.set_ylabel('Photon index')
axO.set_yscale('log'); axO.set_ylabel('Flux density [mJy]')
axO.set_ylim(1e-4,30); axO.legend(fontsize=8)
axO.text(0.03,0.06,'model curves = CORE ONLY\n(wing+flare must supply the rest)',
         transform=axO.transAxes,fontsize=8,style='italic')
fig.suptitle('Single-power-law core FS: three branch solutions vs data',fontweight='bold')
fig.tight_layout()
fig.savefig("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/threecase_lc.png",dpi=150)
print("saved threecase_lc.png")
