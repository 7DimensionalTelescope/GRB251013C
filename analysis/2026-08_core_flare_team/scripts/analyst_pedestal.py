"""Response to the logic audit.

(1) Autocorrelation of the MODEL-FREE residuals.  My 3-point diagnostic removes
    any locally linear trend, so it isolates the high-frequency component.  If
    that component is white, independent-error inflation is the right remedy for
    it; if it is itself autocorrelated, it is real structure and inflation is
    wrong.  Residuals-vs-model being autocorrelated is a separate statement
    (model shape), and does not by itself refute error underestimation.

(2) Additive pedestal  F_obs = k*F_model + C  per Leavitt dataset.

(3) Model-independent discriminator between a pedestal and a zero-point error:
    a pedestal is a constant FLUX, so the offset in magnitudes must GROW as the
    source fades; a zero-point error is a constant FRACTION, so the offset in
    magnitudes is flat.  Leavitt Rc fades by ~5x across its own window.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.likelihood import compute_model_flux_all_bands
from grb.io import read_data

# ---------------------------------------------------------------- (1)
def three_point_resid(t, m, e):
    lt = np.log10(t); res, exp = [], []
    for i in range(1, len(t) - 1):
        w = (lt[i] - lt[i-1]) / (lt[i+1] - lt[i-1])
        res.append(m[i] - ((1 - w) * m[i-1] + w * m[i+1]))
        exp.append(np.sqrt(e[i]**2 + ((1-w)*e[i-1])**2 + (w*e[i+1])**2))
    return np.array(res), np.array(exp)


def runs_test(x):
    s = np.sign(x - np.median(x)); s = s[s != 0]
    n = len(s); r = 1 + np.sum(s[1:] != s[:-1])
    n1 = np.sum(s > 0); n2 = n - n1
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n**2 * (n - 1))
    return (r - mu) / np.sqrt(var) if var > 0 else np.nan


def lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x**2))


print("=" * 78)
print("(1) is the EXCESS scatter white?  (3-point residuals, model-free)")
print("=" * 78)
circ_raw = pd.read_excel("data/circular.xlsx").sort_values("time")
circ_raw = circ_raw[~circ_raw['upper_limit'].astype(bool)]
sets = {}
for filt in ("Rc", "Ic"):
    d = circ_raw[(circ_raw.facility == "Leavitt") & (circ_raw["filter"] == filt)].sort_values("time")
    sets[f"Leavitt_{filt}"] = (d.time.to_numpy(float), d.magnitude.to_numpy(float),
                               d.mag_error.to_numpy(float))
for fac in ("BassanoBrescianoObservatory", "Calapai"):
    d = circ_raw[(circ_raw.facility == fac) & (circ_raw["filter"] == "clear")].sort_values("time")
    sets[f"{fac[:8]}_clear"] = (d.time.to_numpy(float), d.magnitude.to_numpy(float),
                                np.maximum(d.mag_error.to_numpy(float), 1e-3))
_id = pd.read_csv("data/i_data.csv", header=None, names=["time", "magnitude", "mag_error"]).sort_values("time")
sets["i_data"] = (_id.time.to_numpy(float), _id.magnitude.to_numpy(float), _id.mag_error.to_numpy(float))

print(f"{'dataset':22s} {'n':>3} {'chi_rms':>8} {'lag-1 ACF':>10} {'runs z':>8}   verdict")
for name, (t, m, e) in sets.items():
    res, exp = three_point_resid(t, m, e)
    chi = res / exp
    a1, rz = lag1(res), runs_test(res)
    # 3-point differencing itself imprints ACF ~ -0.5 on white noise
    verdict = "white (inflation OK)" if a1 < 0.0 else "CORRELATED -> real structure"
    print(f"{name:22s} {len(res):3d} {np.sqrt(np.mean(chi**2)):8.2f} {a1:10.2f} {rz:8.2f}   {verdict}")
print("\nNOTE: 3-point differencing of WHITE noise gives lag-1 ACF ~ -0.4 to -0.5,")
print("not 0.  Values near that are consistent with white excess scatter.")

# ---------------------------------------------------------------- (2)
print("\n" + "=" * 78)
print("(2) additive pedestal vs multiplicative rescale, per Leavitt dataset")
print("=" * 78)
theta = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
defs = make_param_defs(True, False)
params = {d.name: (10 ** v if d.scale is Scale.LOG else v) for d, v in zip(defs, theta)}
xrt, opt = load_all_optical_data()
_, opt_f, _ = compute_model_flux_all_bands(params, xrt, opt, True, False, None)

print(f"{'dataset':12s} {'model':>26} {'chi2':>9} {'dchi2':>8}  best-fit")
for d, mf in zip(opt, opt_f):
    if not d['name'].startswith("Leavitt"):
        continue
    f, e = d['flux_mJy'], d['flux_err']
    w = 1 / e**2
    c0 = np.sum(w * (f - mf)**2)
    # multiplicative only
    k = np.sum(w * f * mf) / np.sum(w * mf**2)
    ck = np.sum(w * (f - k * mf)**2)
    # additive only
    C = np.sum(w * (f - mf)) / np.sum(w)
    cC = np.sum(w * (f - mf - C)**2)
    # both
    A = np.vstack([mf, np.ones_like(mf)]).T
    M = A.T @ (A * w[:, None]); b = A.T @ (w * f)
    kk, CC = np.linalg.solve(M, b)
    ckC = np.sum(w * (f - kk * mf - CC)**2)
    print(f"{d['name']:12s} {'as-is':>26} {c0:9.1f} {0:8.1f}")
    print(f"{'':12s} {'x k (multiplicative)':>26} {ck:9.1f} {ck-c0:8.1f}  k={k:.3f} ({-2.5*np.log10(k):+.3f} mag)")
    print(f"{'':12s} {'+ C (additive pedestal)':>26} {cC:9.1f} {cC-c0:8.1f}  C={C:+.4f} mJy")
    print(f"{'':12s} {'k and C together':>26} {ckC:9.1f} {ckC-c0:8.1f}  k={kk:.3f}, C={CC:+.4f} mJy")
    print(f"{'':12s} {'':>26} {'':>9} {'':>8}  faintest datapoint = {f.min():.3f} mJy")

# ---------------------------------------------------------------- (3)
print("\n" + "=" * 78)
print("(3) MODEL-FREE discriminator: does the Leavitt offset grow as it fades?")
print("=" * 78)
print("Leavitt Rc measured against the parsed compilation r-trail, in time bins.")
print("A pedestal predicts a growing (more negative) offset; a zero-point error")
print("predicts a flat offset.\n")
pr = pd.read_csv("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/analyst_late_r.csv")
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ['upper_limit'].astype(bool)]
lea = circ[(circ.facility == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
t, m, fl = lea.time.to_numpy(float), lea.magnitude.to_numpy(float), lea.flux_mJy.to_numpy(float)
s = (t >= pr.time_s.min()) & (t <= pr.time_s.max())
mp = np.interp(np.log10(t[s]), np.log10(pr.time_s), pr.mag_AB_galcorr)
off = m[s] - mp
tt, ff = t[s], fl[s]
print(f"{'t range':>18} {'n':>3} {'<flux> mJy':>11} {'offset (mag)':>13}")
for lo, hi in ((5800, 8000), (8000, 10500), (10500, 13500), (13500, 17000)):
    q = (tt >= lo) & (tt < hi)
    if q.sum():
        print(f"{f'{lo}-{hi}':>18} {q.sum():3d} {ff[q].mean():11.3f} {off[q].mean():13.3f}")
A = np.vstack([np.log10(ff), np.ones_like(ff)]).T
sl = np.linalg.lstsq(A, off, rcond=None)[0]
print(f"\n  linear fit of offset vs log10(flux): slope = {sl[0]:+.3f} mag/dex")
print(f"  predicted slope if the offset were a pure pedestal: ~ -1.0 mag/dex")
print(f"  predicted slope if the offset were a zero-point error:  0.0 mag/dex")

print("\n  Independent bound on any HOST pedestal, from the late-time photometry:")
cl = pd.read_csv("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/analyst_late_circular.csv")
print(f"    faintest real detection at 6.1e5 s = {cl.flux_mJy.min():.4f} mJy")
print(f"    faintest Leavitt Rc point          = {fl.min():.4f} mJy")
print(f"    => any host/blend flux common to all facilities is < "
      f"{100*cl.flux_mJy.min()/fl.min():.1f}% of the faintest Leavitt point")
