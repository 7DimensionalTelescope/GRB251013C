"""Leavitt error-budget analysis.

Model-free local scatter: for each interior point i, interpolate its two
neighbours linearly in (log10 t, mag) and take the residual.  This removes any
locally linear trend, so it measures scatter about a smooth light curve without
reference to any physical model.  Error propagation gives the expected residual
sigma from the QUOTED errors; the ratio observed/expected is the inflation
factor needed.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
np.set_printoptions(precision=4, suppress=True)

raw = pd.read_excel("data/circular.xlsx").sort_values("time")
raw = raw[~raw['upper_limit'].astype(bool)]


def three_point(t, m, e, label):
    lt = np.log10(t)
    res, exp = [], []
    for i in range(1, len(t) - 1):
        w = (lt[i] - lt[i-1]) / (lt[i+1] - lt[i-1])
        pred = (1 - w) * m[i-1] + w * m[i+1]
        res.append(m[i] - pred)
        exp.append(np.sqrt(e[i]**2 + ((1-w)*e[i-1])**2 + (w*e[i+1])**2))
    res, exp = np.array(res), np.array(exp)
    rms = np.sqrt(np.mean(res**2))
    # robust version
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    chi = res / exp
    print(f"\n{label}: n={len(t)}, {len(res)} interior residuals")
    print(f"  quoted mag errors: median {np.median(e):.3f}  range {e.min():.3f}-{e.max():.3f}")
    print(f"  residual RMS      = {rms:.4f} mag")
    print(f"  residual MAD-sig  = {mad:.4f} mag")
    print(f"  expected RMS      = {np.sqrt(np.mean(exp**2)):.4f} mag")
    print(f"  chi = res/exp : RMS = {np.sqrt(np.mean(chi**2)):.2f}  "
          f"MAD-sig = {1.4826*np.median(np.abs(chi-np.median(chi))):.2f}")
    # inflation: sigma_new^2 = sigma_quoted^2 + s^2 solved so chi_rms -> 1
    # residual variance predicted with floor s: exp^2 + s^2*(1+(1-w)^2+w^2) ~ exp^2 + 1.5 s^2
    from scipy.optimize import brentq
    def f(s):
        return np.mean(res**2 / (exp**2 + 1.5*s**2)) - 1.0
    if f(0) > 0:
        s = brentq(f, 1e-6, 1.0)
        print(f"  --> extra scatter to add in quadrature: {s:.4f} mag")
        print(f"      implies typical total err {np.sqrt(np.median(e)**2+s**2):.3f} mag "
              f"(inflation x{np.sqrt(np.median(e)**2+s**2)/np.median(e):.2f} on the median point)")
    else:
        print("  --> quoted errors already adequate")
    return res, exp


print("=" * 78)
print("PART 1: model-free local scatter of the fitted Leavitt datasets")
print("=" * 78)
for filt in ("Rc", "Ic"):
    d = raw[(raw['facility'] == 'Leavitt') & (raw['filter'] == filt)].sort_values('time')
    three_point(d['time'].to_numpy(float), d['magnitude'].to_numpy(float),
                d['mag_error'].to_numpy(float), f"Leavitt {filt}")

print("\n" + "=" * 78)
print("PART 2: same estimator on other dense series, for context")
print("=" * 78)
for fac, filt in (('BassanoBrescianoObservatory', 'clear'), ('Calapai', 'clear')):
    d = raw[(raw['facility'] == fac) & (raw['filter'] == filt)].sort_values('time')
    if len(d) > 5:
        three_point(d['time'].to_numpy(float), d['magnitude'].to_numpy(float),
                    np.maximum(d['mag_error'].to_numpy(float), 1e-3), f"{fac} {filt}")

idat = pd.read_csv("data/i_data.csv", header=None, names=["time", "magnitude", "mag_error"])
idat = idat.sort_values("time")
three_point(idat['time'].to_numpy(float), idat['magnitude'].to_numpy(float),
            idat['mag_error'].to_numpy(float), "i_data (flat 0.1 mag errors)")

print("\n" + "=" * 78)
print("PART 3: Leavitt vs independent facilities in the SAME filter (Rc)")
print("=" * 78)
lea = raw[(raw['facility'] == 'Leavitt') & (raw['filter'] == 'Rc')].sort_values('time')
lt, lm = np.log10(lea['time'].to_numpy(float)), lea['magnitude'].to_numpy(float)
others = raw[(raw['facility'] != 'Leavitt') & (raw['filter'] == 'Rc') &
             (raw['time'] > lea['time'].min()) & (raw['time'] < lea['time'].max())]
print(f"{'t':>10} {'facility':36s} {'m_oth':>7} {'err':>6} {'m_Lea(interp)':>14} {'diff':>7}")
diffs = []
for _, r in others.iterrows():
    mi = np.interp(np.log10(r['time']), lt, lm)
    d = r['magnitude'] - mi
    diffs.append((r['facility'], d, r['mag_error']))
    print(f"{r['time']:10.1f} {r['facility'][:36]:36s} {r['magnitude']:7.2f} "
          f"{r['mag_error']:6.2f} {mi:14.3f} {d:+7.3f}")
if diffs:
    dv = np.array([d for _, d, _ in diffs])
    print(f"\n  mean offset (other - Leavitt) = {dv.mean():+.3f} mag, "
          f"scatter {dv.std(ddof=1):.3f}, n={len(dv)}")
    for fac in sorted(set(f for f, _, _ in diffs)):
        sub = np.array([d for f, d, _ in diffs if f == fac])
        print(f"    {fac[:40]:40s} n={len(sub)} mean {sub.mean():+.3f}")
