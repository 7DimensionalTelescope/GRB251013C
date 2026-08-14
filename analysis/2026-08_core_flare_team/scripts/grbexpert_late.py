import numpy as np
d=np.load('sample_parsed.npz')
# actual = plotted - offset ; offsets as plotted in the figure legend
off={'VT/B':+2,'r':0,'R':0,'VT/R':0,'i':-1,'z':-2,'y':-3,'green':+1}
lam={'VT/B':500.,'r':620.,'R':641.,'VT/R':800.,'i':754.,'z':870.,'y':1000.}
edges=np.array([2e5,3e5,4.5e5,7e5,1.0e6,1.4e6,2.2e6])
print("TRUE (de-offset) magnitudes, median per bin  [apparent AB, NOT MW-ext corrected]")
hdr=f"{'t range (s)':>22} " + "".join(f"{b:>9}" for b in ['r','R','i','z','y'])
print(hdr)
tab={}
for b in ['r','R','i','z','y']:
    t=d[b+'_t']; m=d[b+'_m']-off[b]
    tab[b]=(t,m)
for i in range(len(edges)-1):
    lo,hi=edges[i],edges[i+1]
    row=f"{lo:9.3g}-{hi:9.3g} "
    for b in ['r','R','i','z','y']:
        t,m=tab[b]; s=(t>=lo)&(t<hi)
        row+= f"{np.median(m[s]):9.2f}" if s.sum()>2 else f"{'--':>9}"
    print(row)
print()
print("Colors (true, AB):")
print(f"{'t range (s)':>22} {'r-z':>8} {'r-i':>8} {'i-z':>8}   n(r) n(i) n(z)")
for i in range(len(edges)-1):
    lo,hi=edges[i],edges[i+1]
    v={}
    for b in ['r','i','z']:
        t,m=tab[b]; s=(t>=lo)&(t<hi)
        v[b]=np.median(m[s]) if s.sum()>2 else np.nan
        v['n'+b]=s.sum()
    print(f"{lo:9.3g}-{hi:9.3g} {v['r']-v['z']:8.2f} {v['r']-v['i']:8.2f} {v['i']-v['z']:8.2f}   {v['nr']:4d} {v['ni']:4d} {v['nz']:4d}")
print()
print("Decline rate alpha (F ~ t^-alpha) from consecutive bin medians:")
for b in ['r','R','i','z','y']:
    t,m=tab[b]
    cs=[]
    for i in range(len(edges)-1):
        s=(t>=edges[i])&(t<edges[i+1])
        cs.append((np.median(t[s]),np.median(m[s])) if s.sum()>2 else (np.nan,np.nan))
    out=[]
    for i in range(len(cs)-1):
        (t1,m1),(t2,m2)=cs[i],cs[i+1]
        if np.isfinite(m1) and np.isfinite(m2):
            out.append(f"{(m2-m1)/2.5/np.log10(t2/t1):5.2f}")
        else: out.append("   --")
    print(f"  {b:5s}: "+" ".join(out))
