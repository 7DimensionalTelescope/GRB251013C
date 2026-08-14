"""Claim 4, redone with the corrected extraction (no MAXCOMP truncation; z halo-stripped)."""
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
 "i": (R>180)&(G>150)&(B<150)&(R-B>70), "green": (G>60)&(G>R+20)&(G>B+10),
 "VT/R": (R>=70)&(R<=170)&(G<60)&(B<60), "r": (R>170)&(G<90)&(B<90),
 "VT/B": (B>150)&(B>R+60)&(B>G+60),
 "R": (R>180)&(G>80)&(G<175)&(B>80)&(B<175)&(abs(G-B)<45),
 "z": (sat<25)&(mean>110)&(mean<205), "y": (mean<60)&(sat<25),
}
black_halo = ndimage.binary_dilation((sat<25)&(mean<60)&inb, np.ones((3,3)), iterations=2)

def extract(name):
    m = CLASSES[name] & inb
    cap = 10**9
    if name == "z":
        m = m & ~black_halo               # strip antialias halo of black markers/UI button
    if name == "y":
        cap = 3000                        # drop the round UI button (4022 px)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = (sizes >= 3) & (sizes <= cap)
    ys, xs = np.nonzero(m); ok = keep[lab[ys, xs] - 1]
    return 10**(2 + (xs[ok] - X0) / PXDEC), 14 + (ys[ok] - Y0) / PXMAG

def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w/2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz"); tg = d["tgrid"]
WINDOWS = [(2e4, 6e4), (1e5, 4e5), (4e5, 7e5)]
BANDS = ["r", "VT/R", "R", "i", "z", "green", "VT/B", "y"]
tr = {b: binned(*extract(b)) for b in BANDS}

print("=== CORRECTED claim-4 table: median (model - parsed) mag, + = model too faint ===")
print(f"{'':7s}" + "".join(f"{f'[{a:.0e},{b:.0e}]':^26s}" for a, b in WINDOWS))
print(f"{'band':7s}" + "".join(f"{'FLARE-X':>10s}{'INCUMB':>10s}{'nbin':>6s}" for _ in WINDOWS))
score = {w: [0, 0] for w in WINDOWS}
for band in BANDS:
    bt, bm = tr[band]; row = f"{band:7s}"; key = band.replace("/", "")
    for w in WINDOWS:
        s = (bt >= w[0]) & (bt < w[1])
        if s.sum() == 0: row += f"{'-':>10s}{'-':>10s}{0:>6d}"; continue
        f_ = float(np.median(np.interp(np.log10(bt[s]), np.log10(tg), d[f"FLARE-X_{key}"]) - bm[s]))
        i_ = float(np.median(np.interp(np.log10(bt[s]), np.log10(tg), d[f"INCUMBENT_{key}"]) - bm[s]))
        score[w][0 if abs(f_) < abs(i_) else 1] += 1
        row += f"{f_:>10.2f}{i_:>10.2f}{s.sum():>6d}"
    print(row)
print()
for w in WINDOWS:
    print(f"  window [{w[0]:.0e},{w[1]:.0e}]: FLARE-X closer in {score[w][0]} bands, "
          f"INCUMBENT in {score[w][1]}")

print("\n=== aggregate over r,VT/R,R,i,z bins, 2e4 < t < 7e5 ===")
for tag in ("FLARE-X", "INCUMBENT"):
    res = []
    for band in ("r", "VT/R", "R", "i", "z"):
        bt, bm = tr[band]; s = (bt >= 2e4) & (bt < 7e5)
        res.extend((np.interp(np.log10(bt[s]), np.log10(tg), d[f"{tag}_{band.replace('/','')}"]) - bm[s]).tolist())
    a = np.array(res)
    print(f"  {tag:10s} n={len(a):4d} median={np.median(a):+6.2f} mean={a.mean():+6.2f} "
          f"rms={np.sqrt((a**2).mean()):5.2f} median|res|={np.median(np.abs(a)):5.2f}")

np.savez("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_parsed_fixed.npz",
         **{f"{b}_t": extract(b)[0] for b in BANDS}, **{f"{b}_m": extract(b)[1] for b in BANDS})
print("\nwrote /home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_parsed_fixed.npz")
