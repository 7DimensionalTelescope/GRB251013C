"""Provenance of the Leavitt mag_error column.

Real S/N-derived photometric errors scale with source brightness: for a
background/sky-limited detection sigma_mag ~ 10^(0.4 m) (slope 0.4 dex per mag),
for a source-Poisson-limited one ~10^(0.2 m).  Assumed/flat errors show no such
scaling.  Fit log10(sigma) against magnitude and read off the slope.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)

SRC = "data/circular.xlsx"
raw = pd.read_excel(SRC)
print(f"source file: {SRC}   column checked: 'mag_error'")
print(f"columns present: {list(raw.columns)}")
print("(there is no separate systematic/statistical error column, and no")
print(" 'error_type' or provenance column - only the single mag_error)\n")
raw = raw[~raw['upper_limit'].astype(bool)]

print("=" * 78)
print("Leavitt: quoted error vs magnitude")
print("=" * 78)
for filt in ("Rc", "Ic"):
    d = raw[(raw.facility == "Leavitt") & (raw["filter"] == filt)].sort_values("magnitude")
    m, e = d.magnitude.to_numpy(float), d.mag_error.to_numpy(float)
    print(f"\n--- Leavitt {filt} (n={len(d)}) ---")
    print("  distinct quoted errors and the magnitude range each covers:")
    for u in np.unique(e):
        s = e == u
        print(f"    sigma={u:.2f}  n={s.sum():2d}  mag {m[s].min():.2f}-{m[s].max():.2f}")
    sl, ic = np.polyfit(m, np.log10(e), 1)
    pred = 10 ** (ic + sl * m)
    print(f"  fit log10(sigma) = {sl:.3f} * mag + const   (rms resid "
          f"{np.std(np.log10(e) - np.log10(pred)):.3f} dex)")
    print(f"    slope 0.40 = sky/background-limited, 0.20 = source-Poisson-limited,")
    print(f"    slope 0.00 = flat/assumed  ->  measured {sl:.2f}")
    r = np.corrcoef(m, np.log10(e))[0, 1]
    print(f"  correlation of log(sigma) with magnitude: r = {r:.3f}")

print("\n" + "=" * 78)
print("comparison: the same statistic for other facilities in the same file")
print("=" * 78)
print(f"{'facility':38s} {'filt':6s} {'n':>3} {'slope':>7} {'r':>7}  {'distinct sigmas':>16}")
for (fac, filt), d in raw.groupby(["facility", "filter"]):
    e = d.mag_error.to_numpy(float)
    m = d.magnitude.to_numpy(float)
    if len(d) < 6 or np.all(~np.isfinite(e)) or len(np.unique(e)) == 1:
        tag = "FLAT (single value)" if len(np.unique(e)) == 1 and len(d) >= 6 else None
        if tag:
            print(f"{fac[:38]:38s} {str(filt):6s} {len(d):3d} {'-':>7} {'-':>7}  {tag}")
        continue
    ok = np.isfinite(e) & (e > 0)
    if ok.sum() < 6:
        continue
    sl = np.polyfit(m[ok], np.log10(e[ok]), 1)[0]
    r = np.corrcoef(m[ok], np.log10(e[ok]))[0, 1]
    print(f"{fac[:38]:38s} {str(filt):6s} {ok.sum():3d} {sl:7.2f} {r:7.2f}  "
          f"{len(np.unique(e[ok])):>16d}")

print("\n" + "=" * 78)
print("is the EXCESS scatter constant in magnitude, or does it scale with S/N?")
print("=" * 78)
def three_point(t, m, e):
    lt = np.log10(t); res, exp = [], []
    for i in range(1, len(t) - 1):
        w = (lt[i] - lt[i-1]) / (lt[i+1] - lt[i-1])
        res.append(m[i] - ((1 - w) * m[i-1] + w * m[i+1]))
        exp.append(np.sqrt(e[i]**2 + ((1-w)*e[i-1])**2 + (w*e[i+1])**2))
    return np.array(res), np.array(exp), m[1:-1]

for filt in ("Rc", "Ic"):
    d = raw[(raw.facility == "Leavitt") & (raw["filter"] == filt)].sort_values("time")
    res, exp, mm = three_point(d.time.to_numpy(float), d.magnitude.to_numpy(float),
                               d.mag_error.to_numpy(float))
    print(f"\n  Leavitt {filt}: residual scatter split by brightness")
    med = np.median(mm)
    for tag, s in (("brighter half", mm <= med), ("fainter half", mm > med)):
        print(f"    {tag:14s} n={s.sum():2d}  quoted_med={np.median(exp[s]):.3f}  "
              f"|resid| rms={np.sqrt(np.mean(res[s]**2)):.4f}  "
              f"chi_rms={np.sqrt(np.mean((res[s]/exp[s])**2)):.2f}")
