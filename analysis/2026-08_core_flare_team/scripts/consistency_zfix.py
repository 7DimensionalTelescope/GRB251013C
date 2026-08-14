"""Corrected z-band extraction (strip antialias halos of black markers / the UI button)
and the effect on claim 4."""
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

dark = (mean < 60)                     # black markers AND the round UI button
zraw = (sat < 25) & (mean > 110) & (mean < 205) & inb
halo = ndimage.binary_dilation(dark, np.ones((7, 7)))
zclean = zraw & ~halo

def comps(m, maxsize):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = (sizes >= 3) & (sizes <= maxsize)
    ys, xs = np.nonzero(m)
    ok = keep[lab[ys, xs] - 1]
    return xs[ok], ys[ok]

def to_data(xs, ys):
    return 10**(2 + (xs - X0) / PXDEC), 14 + (ys - Y0) / PXMAG

for tag, m in (("z as-parsed", zraw), ("z halo-stripped", zclean)):
    xs, ys = comps(m, 400)
    t, mg = to_data(xs, ys)
    print(f"{tag:18s}: {len(t)} px, t=[{t.min():.3g},{t.max():.3g}]")

xs, ys = comps(zclean, 400); t_c, m_c = to_data(xs, ys)
npz = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
t_o, m_o = npz["z_t"], npz["z_m"]

def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz")
tg = d["tgrid"]
bo_t, bo_m = binned(t_o, m_o)
bc_t, bc_m = binned(t_c, m_c)
print("\n z trail: as-parsed vs halo-stripped, and residuals vs both models")
print(f"{'t':>10s}{'parsed':>9s}{'clean':>9s}{'shift':>8s}"
      f"{'F-X res(old)':>14s}{'F-X res(new)':>14s}{'INC res(old)':>14s}{'INC res(new)':>14s}")
for a, b in [(2e4, 6e4), (6e4, 1e5), (1e5, 4e5), (4e5, 7e5)]:
    so = (bo_t >= a) & (bo_t < b); sc = (bc_t >= a) & (bc_t < b)
    if so.sum() == 0 or sc.sum() == 0:
        print(f"[{a:.0e},{b:.0e}) -- no bins --"); continue
    mo, mc = np.median(bo_m[so]), np.median(bc_m[sc])
    fo = np.median(np.interp(np.log10(bo_t[so]), np.log10(tg), d["FLARE-X_z"]) - bo_m[so])
    fc = np.median(np.interp(np.log10(bc_t[sc]), np.log10(tg), d["FLARE-X_z"]) - bc_m[sc])
    io = np.median(np.interp(np.log10(bo_t[so]), np.log10(tg), d["INCUMBENT_z"]) - bo_m[so])
    ic = np.median(np.interp(np.log10(bc_t[sc]), np.log10(tg), d["INCUMBENT_z"]) - bc_m[sc])
    print(f"[{a:.0e},{b:.0e}) parsed={mo:6.2f} clean={mc:6.2f} shift={mc-mo:+6.2f} | "
          f"F-X {fo:+6.2f} -> {fc:+6.2f} | INC {io:+6.2f} -> {ic:+6.2f}")

# does the halo strip change any other band?
classes = {
 "i":     lambda r,g,b: (r>180)&(g>150)&(b<150)&(r-b>70),
 "green": lambda r,g,b: (g>60)&(g>r+20)&(g>b+10),
 "VT/R":  lambda r,g,b: (r>=70)&(r<=170)&(g<60)&(b<60),
 "r":     lambda r,g,b: (r>170)&(g<90)&(b<90),
 "VT/B":  lambda r,g,b: (b>150)&(b>r+60)&(b>g+60),
 "R":     lambda r,g,b: (r>180)&(g>80)&(g<175)&(b>80)&(b<175)&(abs(g-b)<45),
}
R, G, B = im[...,0], im[...,1], im[...,2]
print("\noverlap of each class with the black-marker halo (contamination risk):")
for name, f in classes.items():
    m = f(R, G, B) & inb
    print(f"  {name:6s}: {m.sum():6d} px, {(m & halo).sum():5d} inside halo "
          f"({(m & halo).sum()/max(m.sum(),1):5.1%})")
