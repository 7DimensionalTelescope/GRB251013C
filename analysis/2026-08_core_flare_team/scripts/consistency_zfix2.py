"""Corrected z extraction: strip only the antialias halo of ACHROMATIC-black markers."""
import numpy as np
from PIL import Image
from scipy import ndimage

im = np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H, W, _ = im.shape
sat, mean = im.max(2) - im.min(2), im.mean(2)
X0, PXDEC, Y0, PXMAG = 168.40, 203.95, 180.60, 55.964
plot_area = np.zeros((H, W), bool); plot_area[100:775, 130:1040] = True
legend = np.zeros((H, W), bool); legend[95:380, 850:1045] = True
inb = plot_area & ~legend

black = (sat < 25) & (mean < 60) & inb        # y markers AND the round UI button
zraw = (sat < 25) & (mean > 110) & (mean < 205) & inb
halo = ndimage.binary_dilation(black, np.ones((3, 3)), iterations=2)

def to_data(xs, ys): return 10**(2 + (xs - X0) / PXDEC), 14 + (ys - Y0) / PXMAG
def comps(m, minsize=3, maxsize=400):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = (sizes >= minsize) & (sizes <= maxsize)
    ys, xs = np.nonzero(m); ok = keep[lab[ys, xs] - 1]
    return xs[ok], ys[ok]

variants = {
    "as-parsed (npz recipe)":       comps(zraw),
    "halo-stripped":                comps(zraw & ~halo),
    "halo-stripped, blob>=10px":    comps(zraw & ~halo, minsize=10),
}
def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz"); tg = d["tgrid"]
print("z-band plotted-mag trail under three extraction variants:")
print(f"{'t bin':>10s}" + "".join(f"{k:>28s}" for k in variants))
bs = {k: binned(*to_data(*v)) for k, v in variants.items()}
allt = sorted(set(np.concatenate([b[0] for b in bs.values()])))
for t in allt:
    if not (2.5e4 <= t <= 1e5): continue
    row = f"{t:10.4g}"
    for k in variants:
        bt, bm = bs[k]
        j = np.argmin(np.abs(bt - t))
        row += f"{(bm[j] if abs(bt[j]-t) < 1e-6 else np.nan):>28.2f}"
    print(row)

print("\nresidual medians in [2e4,6e4] (model - z data):")
for k in variants:
    bt, bm = bs[k]
    s = (bt >= 2e4) & (bt < 6e4)
    if s.sum() == 0:
        print(f"  {k:30s}: no bins"); continue
    f_ = np.median(np.interp(np.log10(bt[s]), np.log10(tg), d["FLARE-X_z"]) - bm[s])
    i_ = np.median(np.interp(np.log10(bt[s]), np.log10(tg), d["INCUMBENT_z"]) - bm[s])
    print(f"  {k:30s}: nbin={s.sum():2d}  FLARE-X={f_:+.2f}  INCUMBENT={i_:+.2f}  "
          f"-> {'FLARE-X' if abs(f_) < abs(i_) else 'INCUMBENT'} closer")

# sanity: other bands unchanged by the black-halo strip?
R, G, B = im[..., 0], im[..., 1], im[..., 2]
others = {
 "i": (R>180)&(G>150)&(B<150)&(R-B>70), "green": (G>60)&(G>R+20)&(G>B+10),
 "VT/R": (R>=70)&(R<=170)&(G<60)&(B<60), "r": (R>170)&(G<90)&(B<90),
 "VT/B": (B>150)&(B>R+60)&(B>G+60),
 "R": (R>180)&(G>80)&(G<175)&(B>80)&(B<175)&(abs(G-B)<45),
}
print("\nother classes overlapping the achromatic-black halo:")
for name, m in others.items():
    m = m & inb
    print(f"  {name:6s}: {(m & halo).sum():5d}/{m.sum():6d} = {(m & halo).sum()/max(m.sum(),1):5.1%}")
