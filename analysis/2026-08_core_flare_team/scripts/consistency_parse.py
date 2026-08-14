"""Claim 3: independent re-extraction of sample.png + spot-checks vs sample_parsed.npz."""
import numpy as np
from PIL import Image
from scipy import ndimage

im = np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H, W, _ = im.shape
r, g, b = im[..., 0], im[..., 1], im[..., 2]
mx, mn = im.max(2), im.min(2)
sat, mean = mx - mn, im.mean(2)

# MY calibration, from the gridline fit (consistency_grid.py)
X0M, PXDECM = 168.40, 203.95
Y0M, PXMAGM = 180.60, 55.964
# CLAIMED calibration
X0C, PXDECC = 168.0, 204.0
Y0C, PXMAGC = 180.0, 56.0

def conv(xs, ys, X0, PXDEC, Y0, PXMAG):
    return 10**(2 + (xs - X0) / PXDEC), 14 + (ys - Y0) / PXMAG

print("=== calibration difference over the plotted range ===")
for x in (168, 372, 576, 780, 984):
    tm = 10**(2 + (x - X0M) / PXDECM); tc = 10**(2 + (x - X0C) / PXDECC)
    print(f"  x={x:4d}: t_mine={tm:10.4g}  t_claim={tc:10.4g}  ratio={tm/tc:.4f}")
for y in (180, 292, 404, 516, 628, 740):
    mm = 14 + (y - Y0M) / PXMAGM; mc = 14 + (y - Y0C) / PXMAGC
    print(f"  y={y:4d}: mag_mine={mm:7.4f}  mag_claim={mc:7.4f}  diff={mm-mc:+.4f} mag")

plot_area = np.zeros((H, W), bool); plot_area[100:775, 130:1040] = True
legend = np.zeros((H, W), bool); legend[95:380, 850:1045] = True
inb = plot_area & ~legend

classes = {
 "i":     (r > 180) & (g > 150) & (b < 150) & (r - b > 70),
 "green": (g > 60) & (g > r + 20) & (g > b + 10),
 "VT/R":  (r >= 70) & (r <= 170) & (g < 60) & (b < 60),
 "r":     (r > 170) & (g < 90) & (b < 90),
 "VT/B":  (b > 150) & (b > r + 60) & (b > g + 60),
 "R":     (r > 180) & (g > 80) & (g < 175) & (b > 80) & (b < 175) & (abs(g - b) < 45),
 "z":     (sat < 25) & (mean > 110) & (mean < 205),
 "y":     (mean < 60) & (sat < 25),
}
MAXCOMP = {"R": 300, "z": 400, "y": 300}

mine = {}
for name, mask in classes.items():
    m = mask & inb
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = (sizes >= 3) & (sizes <= MAXCOMP.get(name, 4000))
    ys, xs = np.nonzero(m)
    ok = keep[lab[ys, xs] - 1]
    mine[name] = (xs[ok], ys[ok])

npz = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")

def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

print("\n=== re-extraction vs stored npz (same masks, my calibration) ===")
print(f"{'band':7s}{'n_mine':>8s}{'n_npz':>8s}{'max|dmag|':>11s}{'med|dmag|':>11s}{'max|dlogt|':>12s}")
for name in classes:
    xs, ys = mine[name]
    tm, mm = conv(xs, ys, X0M, PXDECM, Y0M, PXMAGM)
    tn, mn_ = npz[f"{name}_t"], npz[f"{name}_m"]
    if len(tm) != len(tn):
        print(f"{name:7s}{len(tm):>8d}{len(tn):>8d}   *** pixel-count mismatch ***")
        continue
    o1 = np.lexsort((mm, tm)); o2 = np.lexsort((mn_, tn))
    dm = np.abs(mm[o1] - mn_[o2]); dt = np.abs(np.log10(tm[o1]) - np.log10(tn[o2]))
    print(f"{name:7s}{len(tm):>8d}{len(tn):>8d}{dm.max():>11.4f}{np.median(dm):>11.4f}{dt.max():>12.5f}")

# ---------- feature spot-checks, read straight off the pixels ----------
print("\n=== spot-checks (my calibration, PLOTTED mags incl. legend offsets) ===")

def band_pts(name):
    xs, ys = mine[name]
    return conv(xs, ys, X0M, PXDECM, Y0M, PXMAGM)

def brightest_in(name, t1, t2):
    t, m = band_pts(name)
    s = (t >= t1) & (t < t2)
    if s.sum() == 0: return None
    i = np.argmin(m[s])          # smallest mag = brightest
    return t[s][i], m[s][i]

def faintest_in(name, t1, t2):
    t, m = band_pts(name)
    s = (t >= t1) & (t < t2)
    i = np.argmax(m[s])
    return t[s][i], m[s][i]

def npz_binned_at(name, t1, t2):
    bt, bm = binned(npz[f"{name}_t"], npz[f"{name}_m"])
    s = (bt >= t1) & (bt < t2)
    return bt[s], bm[s]

# 1. i-band (yellow, plotted = i-1) flare/bump peak, 2e3-5e3 s
for name, lo, hi, what in [("i", 2e3, 5e3, "i flare-bump peak"),
                           ("VT/R", 1.8e3, 4e3, "VT/R flare-bump peak"),
                           ("VT/B", 1.8e3, 4e3, "VT/B flare-bump peak")]:
    bt, bm = binned(*band_pts(name))
    s = (bt >= lo) & (bt < hi)
    j = np.argmin(bm[s])
    nbt, nbm = npz_binned_at(name, lo, hi)
    k = np.argmin(nbm)
    print(f"  {what:24s} mine t={bt[s][j]:8.0f}s m={bm[s][j]:6.3f} | "
          f"npz t={nbt[k]:8.0f}s m={nbm[k]:6.3f}")

# pre-bump minimum (the dip)
for name in ("i", "VT/R", "VT/B"):
    bt, bm = binned(*band_pts(name))
    s = (bt >= 1e3) & (bt < 2e3)
    j = np.argmax(bm[s])
    print(f"  {name+' pre-bump dip':24s} mine t={bt[s][j]:8.0f}s m={bm[s][j]:6.3f}")

# 2. earliest yellow point
t, m = band_pts("i")
i0 = np.argmin(t)
tn, mn2 = npz["i_t"], npz["i_m"]
j0 = np.argmin(tn)
print(f"  {'earliest i pixel':24s} mine t={t[i0]:8.1f}s m={m[i0]:6.3f} | "
      f"npz t={tn[j0]:8.1f}s m={mn2[j0]:6.3f}")
s = t < 130
print(f"  {'i at t<130 s':24s} mine n={s.sum():4d} m range [{m[s].min():.3f},{m[s].max():.3f}]")

# 3. late z plateau
for lo, hi in [(2.5e5, 5e5), (5e5, 1.1e6), (1.1e6, 2.2e6)]:
    bt, bm = binned(*band_pts("z"))
    s = (bt >= lo) & (bt < hi)
    if s.sum():
        print(f"  z plotted mag in [{lo:.1e},{hi:.1e}]: mine median={np.median(bm[s]):6.3f} "
              f"(n_bins={s.sum()}) -> true z = {np.median(bm[s])+2:6.3f}")

# 4. y-band earliest clump
t, m = band_pts("y")
print(f"  y-band: n={len(t)} t=[{t.min():.3g},{t.max():.3g}] "
      f"m=[{m.min():.2f},{m.max():.2f}]")

# 5. r-band at 1e5 and 3e5
for name in ("r", "VT/R", "green", "z"):
    bt, bm = binned(*band_pts(name))
    for tt in (1e5, 3e5, 6e5):
        j = np.argmin(np.abs(np.log10(bt) - np.log10(tt)))
        if abs(np.log10(bt[j] / tt)) < 0.1:
            print(f"  {name:6s} at t~{tt:.0e}: plotted m={bm[j]:6.3f} (bin t={bt[j]:.3g})")

# 6. sanity: how thick is a data point (parse error in px)?
for name in ("i", "r", "z"):
    xs, ys = mine[name]
    lab, n = ndimage.label(classes[name] & inb, structure=np.ones((3, 3)))
    sizes = ndimage.sum(classes[name] & inb, lab, range(1, n + 1))
    keep = (sizes >= 3) & (sizes <= MAXCOMP.get(name, 4000))
    ss = sizes[keep]
    print(f"  {name:6s} marker blobs: n={keep.sum()} size median={np.median(ss):.0f} px "
          f"=> radius ~{np.sqrt(np.median(ss)/np.pi):.1f} px = {np.sqrt(np.median(ss)/np.pi)/PXMAGM:.3f} mag")
