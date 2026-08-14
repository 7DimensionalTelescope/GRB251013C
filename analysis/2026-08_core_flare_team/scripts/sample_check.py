"""Parse data points from sample.png (multi-band visible LC compilation) and
overlay the current best-fit model (INITIAL_GUESS, logP=-436.65).

Calibration (from gridlines): x=168->1e2 s, 204 px/decade; y=180->mag14, 56 px/mag.
Legend offsets: plotted = mag + offset, offsets: VT/B +2, r 0, R 0, VT/R 0,
i -1, I -1, z -2, y -3. Green band: legend cut off; assumed g with offset +1
(flagged as a guess).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

im=np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H,W,_=im.shape
r,g,b=im[...,0],im[...,1],im[...,2]
mx=im.max(2); mn=im.min(2); sat=mx-mn; mean=im.mean(2)

X0,PXDEC=168.0,204.0
Y0,PXMAG=180.0,56.0
def to_data(xs,ys):
    return 10**(2+(xs-X0)/PXDEC), 14+(ys-Y0)/PXMAG

plot_area=np.zeros((H,W),bool); plot_area[100:775,130:1040]=True
legend=np.zeros((H,W),bool); legend[95:380,850:1045]=True
inb=plot_area&~legend

classes={
 "i":      (r>180)&(g>150)&(b<150)&(r-b>70),
 "green":  (g>60)&(g>r+20)&(g>b+10),
 "VT/R":   (r>=70)&(r<=170)&(g<60)&(b<60),
 "r":      (r>170)&(g<90)&(b<90),
 "VT/B":   (b>150)&(b>r+60)&(b>g+60),
 "R":      (r>180)&(g>80)&(g<175)&(b>80)&(b<175)&(abs(g-b)<45),
 "z":      (sat<25)&(mean>110)&(mean<205),
 "y":      (mean<60)&(sat<25),
}
MAXCOMP={"R":300,"z":400,"y":300}   # kill watermark letters / frame / text
pts={}
for name,mask in classes.items():
    m=mask&inb
    lab,n=ndimage.label(m, structure=np.ones((3,3)))
    sizes=ndimage.sum(m,lab,range(1,n+1))
    keep=np.ones(n,bool)
    keep&=(sizes>=3)
    cap=MAXCOMP.get(name,4000)
    keep&=(sizes<=cap)
    ys,xs=np.nonzero(m)
    l=lab[ys,xs]
    ok=keep[l-1]
    t,mag=to_data(xs[ok],ys[ok])
    pts[name]=(t,mag)
    print(f"{name:6s}: {ok.sum():6d} px in {int(keep.sum())} comps")

# binned trails: median plotted mag in 0.04-dex time bins
def binned(t,mag,w=0.04):
    lt=np.log10(t); out=[]
    for c in np.arange(2.0,6.4,w):
        s=(lt>=c)&(lt<c+w)
        if s.sum()>=3: out.append((10**(c+w/2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2,0))

OFFSET={"VT/B":+2,"r":0,"R":0,"VT/R":0,"i":-1,"I":-1,"z":-2,"y":-3,"green":+1}
WAVE_AA={"VT/B":4450,"green":4770,"r":6215,"R":6580,"VT/R":6580,"i":7545,"z":8700,"y":9620}

# ---- model curves ----
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction

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
C_AA=2.99792458e18
tgrid=np.geomspace(1e2,2.5e6,300)
model_mag={}
for band,wl in WAVE_AA.items():
    nu=C_AA/wl
    comp=compute_model_components(GUESS,tgrid,nu,None,True,True)
    F=comp['total']  # mJy, host-extincted, galactic-corrected frame
    A_gal=float(galactic_extinction(np.array([wl]))[0])
    mag=-2.5*np.log10(F)+16.4+A_gal      # apparent AB
    model_mag[band]=mag+OFFSET[band]     # plotted convention
    print(f"{band:6s} lambda={wl}A A_gal={A_gal:.3f}")

# ---- overlay figure ----
COL={"i":"gold","green":"green","VT/R":"darkred","r":"red","VT/B":"blue","R":"salmon","z":"gray","y":"black"}
fig,ax=plt.subplots(figsize=(11,8))
for name,(t,mag) in pts.items():
    if len(t)==0: continue
    ax.scatter(t,mag,s=2,alpha=0.25,color=COL[name])
    bt,bm=binned(t,mag)
    if len(bt): ax.plot(bt,bm,'.',ms=6,color=COL[name])
for band in WAVE_AA:
    ax.plot(tgrid,model_mag[band],'-',lw=1.6,color=COL[band],
            label=f"model {band}{'+' if OFFSET[band]>=0 else ''}{OFFSET[band]}"+(" (g? guess)" if band=="green" else ""))
ax.set_xscale('log'); ax.invert_yaxis(); ax.set_ylim(25,12.5)
ax.set_xlabel('Time since trigger [s]'); ax.set_ylabel('Plotted AB mag (with offsets)')
ax.legend(fontsize=8,ncol=2); ax.grid(alpha=0.3)
ax.set_title('sample.png parsed points vs current best-fit model')
out="/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/sample_vs_model.png"
fig.tight_layout(); fig.savefig(out,dpi=150); print("saved",out)

# ---- numeric consistency: model minus data (plotted mags) in time windows ----
print("\nresiduals (model - parsed median, mag; +ve = model FAINTER):")
WIN=[(2e2,1e3),(1e3,4e3),(4e3,3e4),(3e4,1e5),(1e5,4e5),(4e5,2e6)]
hdr="band   "+"".join([f"{f'{a:.0e}-{b:.0e}':>16}" for a,b in WIN])
print(hdr)
for band in ("i","r","VT/R","VT/B","z","R","y","green"):
    t,mag=pts[band]
    row=f"{band:7s}"
    bt,bm=binned(t,mag)
    for a,bnd in WIN:
        s=(bt>=a)&(bt<bnd)
        if s.sum()==0: row+=f"{'-':>16}"; continue
        mm=np.interp(np.log10(bt[s]),np.log10(tgrid),model_mag[band]) if band in model_mag else np.nan
        row+=f"{np.median(mm-bm[s]):>16.2f}"
    print(row)
np.savez("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz",
         **{f"{k}_t":v[0] for k,v in pts.items()}, **{f"{k}_m":v[1] for k,v in pts.items()})
