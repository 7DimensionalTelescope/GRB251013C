"""Finer time-resolved residuals: where does the late optical excess actually start?"""
import numpy as np
d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz")
npz = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
tgrid = d["tgrid"]

def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

BANDS = ["r", "VT/R", "R", "z", "green", "VT/B", "i"]
EDGES = [2e4, 4e4, 6e4, 1e5, 1.6e5, 2.5e5, 4e5, 6e5, 8e5, 1.2e6, 2.2e6]
print("median (model - parsed) mag, POSITIVE = model too faint.  F=FLARE-X  I=INCUMBENT")
print(f"{'window':>22s} " + "".join(f"{b:>13s}" for b in BANDS) + f"{'ALL F':>9s}{'ALL I':>9s}")
for a, b in zip(EDGES[:-1], EDGES[1:]):
    rowF, rowI, allF, allI = "", "", [], []
    for band in BANDS:
        bt, bm = binned(npz[f"{band}_t"], npz[f"{band}_m"])
        s = (bt >= a) & (bt < b)
        if s.sum() == 0:
            rowF += f"{'  -  /  -  ':>13s}"; continue
        key = band.replace("/", "")
        rF = np.median(np.interp(np.log10(bt[s]), np.log10(tgrid), d[f"FLARE-X_{key}"]) - bm[s])
        rI = np.median(np.interp(np.log10(bt[s]), np.log10(tgrid), d[f"INCUMBENT_{key}"]) - bm[s])
        rowF += f"{rF:>+6.2f}/{rI:>+6.2f}"
        allF.append(rF); allI.append(rI)
    mF = np.median(allF) if allF else np.nan
    mI = np.median(allI) if allI else np.nan
    print(f"{f'[{a:.1e},{b:.1e})':>22s} " + rowF + f"{mF:>+9.2f}{mI:>+9.2f}")

# z-band anomaly at 2e4-6e4: is the gray trail there self-consistent?
print("\n=== z (gray) trail, bin by bin, plotted mag (offset -2 => true z = m+2) ===")
bt, bm = binned(npz["z_t"], npz["z_m"])
s = (bt >= 2e4) & (bt < 1e5)
for t, m in zip(bt[s], bm[s]):
    fx = np.interp(np.log10(t), np.log10(tgrid), d["FLARE-X_z"])
    ic = np.interp(np.log10(t), np.log10(tgrid), d["INCUMBENT_z"])
    print(f"   t={t:9.3g}  parsed={m:6.2f}  FLARE-X={fx:6.2f}  INCUMB={ic:6.2f}")

# colour check: is the model's optical COLOUR right, independent of normalisation?
print("\n=== colours (plotted-mag differences), data vs models ===")
def val(band, t):
    bt, bm = binned(npz[f"{band}_t"], npz[f"{band}_m"])
    j = np.argmin(np.abs(np.log10(bt) - np.log10(t)))
    if abs(np.log10(bt[j] / t)) > 0.06: return None, None
    key = band.replace("/", "")
    return bm[j], (np.interp(np.log10(bt[j]), np.log10(tgrid), d["FLARE-X_" + key]),
                   np.interp(np.log10(bt[j]), np.log10(tgrid), d["INCUMBENT_" + key]))
for t in (3e4, 5e4, 1.2e5, 3e5, 5.5e5):
    print(f"  t={t:.1e}")
    for pair in [("r", "z"), ("r", "green"), ("r", "VT/B"), ("z", "y"), ("i", "z")]:
        a, mA = val(pair[0], t); b, mB = val(pair[1], t)
        if a is None or b is None: continue
        print(f"     {pair[0]:5s}-{pair[1]:5s}  data={a-b:+6.2f}  "
              f"FLARE-X={mA[0]-mB[0]:+6.2f}  INCUMB={mA[1]-mB[1]:+6.2f}")
