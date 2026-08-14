"""Claim 3: independently re-derive the sample.png pixel->data calibration from gridlines."""
import numpy as np
from PIL import Image

im = np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H, W, _ = im.shape
print("image", W, "x", H)

# Gridlines are faint light-gray lines on white inside the axes.
# Restrict to a band that avoids data, watermark and legend.
sub = im[105:770, 130:1045]
gray = sub.mean(2)
sat = sub.max(2) - sub.min(2)

# --- vertical gridlines: columns that are systematically slightly darker than white
colscore = ((gray < 249) & (gray > 200) & (sat < 12)).mean(0)
cand = np.where(colscore > 0.55)[0]
groups, cur = [], [cand[0]]
for c in cand[1:]:
    if c - cur[-1] <= 2:
        cur.append(c)
    else:
        groups.append(cur); cur = [c]
groups.append(cur)
vx = np.array([np.mean(g) + 130 for g in groups])
print("\nvertical gridline x (px):", np.round(vx, 2))
print("spacings:", np.round(np.diff(vx), 2))

# --- horizontal gridlines
rowscore = ((gray < 249) & (gray > 200) & (sat < 12)).mean(1)
cand = np.where(rowscore > 0.55)[0]
groups, cur = [], [cand[0]]
for c in cand[1:]:
    if c - cur[-1] <= 2:
        cur.append(c)
    else:
        groups.append(cur); cur = [c]
groups.append(cur)
hy = np.array([np.mean(g) + 105 for g in groups])
print("\nhorizontal gridline y (px):", np.round(hy, 2))
print("spacings:", np.round(np.diff(hy), 2))

# --- fit calibration: major x gridlines are decades 1e2..1e6, major y are mags 14..24 step 2
if len(vx) >= 5:
    # keep only the widely-spaced (major/decade) lines
    dec = np.arange(len(vx))
    A = np.polyfit(dec, vx, 1)
    print(f"\nx: px/decade = {A[0]:.3f}, x(first line) = {A[1]:.2f}")
if len(hy) >= 6:
    idx = np.arange(len(hy))
    B = np.polyfit(idx, hy, 1)
    print(f"y: px per 2 mag = {B[0]:.3f}  => px/mag = {B[0]/2:.3f}, y(first) = {B[1]:.2f}")

# --- independent check: tick-label glyph centroids on the axes
# x tick labels sit below y~785; y tick labels left of x~130
def dark_blobs(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 15: continue
        out.append((xs.mean(), ys.mean(), len(ys), xs.min(), xs.max(), ys.min(), ys.max()))
    return out

dk = im.mean(2) < 120
xt = np.zeros_like(dk); xt[786:830, 120:1060] = dk[786:830, 120:1060]
yt = np.zeros_like(dk); yt[100:780, 40:125] = dk[100:780, 40:125]
print("\nx tick-label blobs (x_c, y_c, npx, xmin, xmax):")
for b in sorted(dark_blobs(xt)): print("   ", np.round(b, 1))
print("\ny tick-label blobs (x_c, y_c, npx):")
for b in sorted(dark_blobs(yt), key=lambda t: t[1]): print("   ", np.round(b[:3], 1))
