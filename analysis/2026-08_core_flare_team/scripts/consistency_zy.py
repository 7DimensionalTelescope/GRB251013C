"""Is the 'z' gray trail at 3-6e4 s a real trail, or antialias halo of the black y markers?"""
import numpy as np
from PIL import Image
from scipy import ndimage

im = np.asarray(Image.open("/data/dtak/research/grb/GRB251013C/sample.png").convert("RGB")).astype(int)
H, W, _ = im.shape
mx, mn = im.max(2), im.min(2)
sat, mean = mx - mn, im.mean(2)
X0, PXDEC, Y0, PXMAG = 168.40, 203.95, 180.60, 55.964

def px_of_t(t): return X0 + PXDEC * (np.log10(t) - 2)
def px_of_m(m): return Y0 + PXMAG * (m - 14)

x1, x2 = int(px_of_t(2.8e4)), int(px_of_t(6.5e4))
y1, y2 = int(px_of_m(14.5)), int(px_of_m(19.5))
print(f"crop x=[{x1},{x2}] y=[{y1},{y2}]  (t=2.8e4-6.5e4 s, plotted mag 14.5-19.5)")

crop = im[y1:y2, x1:x2]
sub_mean = crop.mean(2); sub_sat = crop.max(2) - crop.min(2)
zmask = (sub_sat < 25) & (sub_mean > 110) & (sub_mean < 205)
ymask = (sub_sat < 25) & (sub_mean < 60)
print(f"z-class px in crop: {zmask.sum()}   y-class px in crop: {ymask.sum()}")

# For each column, list the vertical runs of z and y pixels
print("\ncolumn-by-column (every 8 px): y-runs and z-runs in plotted mag")
for cx in range(0, x2 - x1, 8):
    yr = np.nonzero(ymask[:, cx])[0]
    zr = np.nonzero(zmask[:, cx])[0]
    t = 10**(2 + (cx + x1 - X0) / PXDEC)
    def runs(a):
        if len(a) == 0: return []
        br = np.nonzero(np.diff(a) > 2)[0]
        segs = np.split(a, br + 1)
        return [(14 + (s[0] + y1 - Y0) / PXMAG, 14 + (s[-1] + y1 - Y0) / PXMAG) for s in segs]
    ys, zs = runs(yr), runs(zr)
    print(f"  t={t:8.3g}  y:{[f'{a:.2f}-{b:.2f}' for a,b in ys]}  z:{[f'{a:.2f}-{b:.2f}' for a,b in zs]}")

# Are z pixels adjacent to y pixels (halo) or spatially separate?
ydil = ndimage.binary_dilation(ymask, np.ones((5, 5)))
touching = (zmask & ydil).sum()
print(f"\nz pixels within 2 px of a y pixel: {touching} / {zmask.sum()} = {touching/max(zmask.sum(),1):.1%}")

# same test in a region where z is unambiguously its own trail (t > 3e5)
x3, x4 = int(px_of_t(3e5)), int(px_of_t(1.5e6))
y3, y4 = int(px_of_m(17.0)), int(px_of_m(21.0))
c2 = im[y3:y4, x3:x4]
m2, s2 = c2.mean(2), c2.max(2) - c2.min(2)
z2 = (s2 < 25) & (m2 > 110) & (m2 < 205); y2m = (s2 < 25) & (m2 < 60)
t2 = (z2 & ndimage.binary_dilation(y2m, np.ones((5, 5)))).sum()
print(f"control (t=3e5-1.5e6): z px={z2.sum()} y px={y2m.sum()}  "
      f"z near y: {t2}/{max(z2.sum(),1)} = {t2/max(z2.sum(),1):.1%}")
