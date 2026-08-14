"""Definitive re-parse: quantify how much the MAXCOMP blob-size cap throws away,
recover the real z trail, and redo claim 4 with corrected trails."""
import numpy as np
from PIL import Image
from scipy import ndimage

im = np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H, W, _ = im.shape
R, G, B = im[..., 0], im[..., 1], im[..., 2]
sat, mean = im.max(2) - im.min(2), im.mean(2)
X0, PXDEC, Y0, PXMAG = 168.40, 203.95, 180.60, 55.964
plot_area = np.zeros((H, W), bool); plot_area[100:775, 130:1040] = True
legend = np.zeros((H, W), bool); legend[95:380, 850:1045] = True
inb = plot_area & ~legend

CLASSES = {
 "i":     (R>180)&(G>150)&(B<150)&(R-B>70),
 "green": (G>60)&(G>R+20)&(G>B+10),
 "VT/R":  (R>=70)&(R<=170)&(G<60)&(B<60),
 "r":     (R>170)&(G<90)&(B<90),
 "VT/B":  (B>150)&(B>R+60)&(B>G+60),
 "R":     (R>180)&(G>80)&(G<175)&(B>80)&(B<175)&(abs(G-B)<45),
 "z":     (sat<25)&(mean>110)&(mean<205),
 "y":     (mean<60)&(sat<25),
}
MAXCOMP = {"R": 300, "z": 400, "y": 300}
black_halo = ndimage.binary_dilation((sat < 25) & (mean < 60) & inb, np.ones((3, 3)), iterations=2)

def to_data(xs, ys): return 10**(2 + (xs - X0) / PXDEC), 14 + (ys - Y0) / PXMAG

print("=== effect of the MAXCOMP blob-size cap in the npz recipe ===")
print(f"{'band':7s}{'cap':>7s}{'n_comp':>8s}{'kept':>7s}{'px_kept':>9s}{'px_dropped_by_cap':>20s}")
dropped_info = {}
for name, m0 in CLASSES.items():
    m = m0 & inb
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    cap = MAXCOMP.get(name, 4000)
    big = (sizes > cap) & (sizes >= 3)
    keep = (sizes >= 3) & (sizes <= cap)
    ys, xs = np.nonzero(m); l = lab[ys, xs]
    px_big = np.isin(l, np.nonzero(big)[0] + 1)
    print(f"{name:7s}{cap:>7d}{n:>8d}{int(keep.sum()):>7d}{int(keep[l-1].sum()):>9d}"
          f"{int(px_big.sum()):>20d}")
    if px_big.sum():
        t, mg = to_data(xs[px_big], ys[px_big])
        dropped_info[name] = (t, mg, sizes[big])

for name, (t, mg, sz) in dropped_info.items():
    print(f"   {name}: dropped blobs sizes={sorted(sz.astype(int))[:8]}... "
          f"spanning t=[{t.min():.3g},{t.max():.3g}] mag=[{mg.min():.2f},{mg.max():.2f}]")

# ---- corrected extraction: no size cap, strip achromatic-black halo from z ----
def extract(name, strip_black_halo=False, cap=10**9):
    m = CLASSES[name] & inb
    if strip_black_halo: m = m & ~black_halo
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = (sizes >= 3) & (sizes <= cap)
    ys, xs = np.nonzero(m); ok = keep[lab[ys, xs] - 1]
    return to_data(xs[ok], ys[ok])

def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

# the round UI button: huge achromatic blob; exclude by removing very large z blobs
# that are far from any other data -- simplest: keep z blobs <= 20000 px and strip halo
z_t, z_m = extract("z", strip_black_halo=True, cap=20000)
print(f"\ncorrected z: {len(z_t)} px, t=[{z_t.min():.3g},{z_t.max():.3g}]")

npz = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz"); tg = d["tgrid"]
bt_old, bm_old = binned(npz["z_t"], npz["z_m"])
bt_new, bm_new = binned(z_t, z_m)
yt, ym = extract("y", cap=20000)
bt_y, bm_y = binned(yt, ym)
print("\nz trail: npz vs corrected, with the y trail for reference (plotted mags)")
print(f"{'t':>10s}{'z_npz':>9s}{'z_fixed':>9s}{'y':>9s}{'F-X model z':>13s}{'INC model z':>13s}")
for t in bt_new:
    if t > 1.2e5: break
    def at(bt, bm):
        j = np.argmin(np.abs(bt - t)); return bm[j] if abs(bt[j] - t) < 1e-6 else np.nan
    print(f"{t:10.4g}{at(bt_old,bm_old):9.2f}{at(bt_new,bm_new):9.2f}{at(bt_y,bm_y):9.2f}"
          f"{np.interp(np.log10(t),np.log10(tg),d['FLARE-X_z']):13.2f}"
          f"{np.interp(np.log10(t),np.log10(tg),d['INCUMBENT_z']):13.2f}")

print("\n=== claim-4 z residuals, corrected ===")
for a, b in [(2e4, 6e4), (1e5, 4e5), (4e5, 7e5)]:
    for tag, bt, bm in (("npz", bt_old, bm_old), ("fixed", bt_new, bm_new)):
        s = (bt >= a) & (bt < b)
        if s.sum() == 0:
            print(f"  [{a:.0e},{b:.0e}) {tag:6s}: no bins"); continue
        f_ = np.median(np.interp(np.log10(bt[s]), np.log10(tg), d["FLARE-X_z"]) - bm[s])
        i_ = np.median(np.interp(np.log10(bt[s]), np.log10(tg), d["INCUMBENT_z"]) - bm[s])
        print(f"  [{a:.0e},{b:.0e}) {tag:6s}: nbin={s.sum():2d} FLARE-X={f_:+.2f} "
              f"INCUMBENT={i_:+.2f} -> {'FLARE-X' if abs(f_)<abs(i_) else 'INCUMBENT'} closer")

# z-y colour with the corrected trail
print("\nz-y plotted colour (offsets z-2, y-3 => true z-y = plotted diff - 1):")
for t in bt_y:
    j = np.argmin(np.abs(bt_new - t))
    if abs(np.log10(bt_new[j] / t)) > 0.03: continue
    print(f"   t={t:9.4g}  z_fixed-y = {bm_new[j]-bm_y[np.argmin(np.abs(bt_y-t))]:+5.2f} "
          f"=> true z-y = {bm_new[j]-bm_y[np.argmin(np.abs(bt_y-t))]-1:+5.2f}  "
          f"(FLARE-X true z-y = {np.interp(np.log10(t),np.log10(tg),d['FLARE-X_z'])-np.interp(np.log10(t),np.log10(tg),d['FLARE-X_y'])-1:+5.2f})")
