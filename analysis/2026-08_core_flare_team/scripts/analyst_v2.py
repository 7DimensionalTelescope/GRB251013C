"""Re-run the i-band diagnostics against the NEW data/i_data.csv (main checkout,
real per-point errors), replicating main's loader transformation but keeping all
other code on the worktree package so only the i-band data changes.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
MAIN = "/data/dtak/research/grb/GRB251013C"
sys.path.insert(0, WT); os.chdir(WT)
from astropy.time import Time
from grb.const import TRIGGER_TIME
from grb.extinction import galactic_extinction
from grb.utils import filter_to_wavelength
from grb.io import read_data

LN = np.log(10) * 0.4
ZP_AB = 3631.0

# ---- load the NEW i_data the way main's io.py does ----
new = pd.read_csv(f"{MAIN}/data/i_data.csv")
trigger_mjd = Time(TRIGGER_TIME.strftime("%Y-%m-%dT%H:%M:%S")).mjd
new["time"] = (new["mjd"].to_numpy(float) - trigger_mjd) * 86400.0
new = new.rename(columns={"mag": "magnitude", "magerr": "mag_error",
                          "instrument_name": "facility"})
new["filter"] = new["filter"].astype(str).str.strip().str.replace("^sdss", "", regex=True)
new = new.sort_values("time").reset_index(drop=True)
wl_i = filter_to_wavelength("i")
a_gal = float(galactic_extinction(np.array([wl_i]))[0])
new["mag_corr"] = new["magnitude"] - a_gal
new["flux_mJy"] = ZP_AB * 10 ** (-0.4 * new["mag_corr"]) * 1e3
new["flux_err"] = new["flux_mJy"] * LN * new["mag_error"]

old = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)

print("=" * 78)
print("i_data: OLD (worktree) vs NEW (main)")
print("=" * 78)
print(f"n: old {len(old)}  new {len(new)}")
print(f"facilities in new file: {sorted(new['facility'].unique())}")
print(f"filters in new file:    {sorted(new['filter'].unique())}")
print(f"time span: old {old['time'].min():.2f}-{old['time'].max():.1f} s   "
      f"new {new['time'].min():.2f}-{new['time'].max():.1f} s")
print(f"mag_error: old {old['mag_error'].min():.3f}-{old['mag_error'].max():.3f} "
      f"(median {old['mag_error'].median():.3f})   "
      f"new {new['mag_error'].min():.3f}-{new['mag_error'].max():.3f} "
      f"(median {new['mag_error'].median():.3f})")
if len(old) == len(new):
    dt = new["time"].to_numpy() - old["time"].to_numpy()
    dm = new["mag_corr"].to_numpy() - old["magnitude"].to_numpy()
    print(f"time  diff: mean {dt.mean():+.3f} s, max |diff| {np.abs(dt).max():.3f} s")
    print(f"mag   diff: mean {dm.mean():+.4f}, max |diff| {np.abs(dm).max():.4f} mag, "
          f"rms {dm.std():.4f}")

# ---- 3-point scatter diagnostic on the new errors ----
def three_point(t, m, e, label):
    lt = np.log10(t); res, exp = [], []
    for i in range(1, len(t) - 1):
        w = (lt[i] - lt[i-1]) / (lt[i+1] - lt[i-1])
        res.append(m[i] - ((1 - w) * m[i-1] + w * m[i+1]))
        exp.append(np.sqrt(e[i]**2 + ((1-w)*e[i-1])**2 + (w*e[i+1])**2))
    res, exp = np.array(res), np.array(exp)
    chi = res / exp
    rms = np.sqrt(np.mean(chi**2))
    from scipy.optimize import brentq
    def f(s):
        return np.mean(res**2 / (exp**2 + 1.5*s**2)) - 1.0
    extra = brentq(f, 1e-6, 1.0) if f(0) > 0 else 0.0
    med = np.median(e)
    print(f"{label:34s} n={len(res):3d} quoted_med={med:.3f}  "
          f"scatter_rms={np.sqrt(np.mean(res**2)):.4f}  chi_rms={rms:.2f}  "
          f"extra={extra:.4f}  -> total {np.sqrt(med**2+extra**2):.3f} "
          f"(x{np.sqrt(med**2+extra**2)/med:.2f})")
    return extra

print("\n" + "=" * 78)
print("model-free local scatter, i-band OLD vs NEW errors")
print("=" * 78)
three_point(old["time"].to_numpy(float), old["magnitude"].to_numpy(float),
            old["mag_error"].to_numpy(float), "i_data OLD (flat 0.100 mag)")
three_point(new["time"].to_numpy(float), new["mag_corr"].to_numpy(float),
            new["mag_error"].to_numpy(float), "i_data NEW (real per-point errors)")

# ---- weight shares with the new i errors ----
print("\n" + "=" * 78)
print("optical statistical weight share, recomputed with the NEW i-band errors")
print("=" * 78)
from VegasAfterglow import Scale
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.likelihood import compute_model_flux_all_bands
theta = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
defs = make_param_defs(True, False)
params = {d.name: (10 ** v if d.scale is Scale.LOG else v) for d, v in zip(defs, theta)}
xrt, opt = load_all_optical_data()
# swap in the new i-band data
for d in opt:
    if d['name'] == 'i-band':
        d['time'] = new['time'].to_numpy(float)
        d['flux_mJy'] = new['flux_mJy'].to_numpy(float)
        d['flux_err'] = new['flux_err'].to_numpy(float)
_, opt_f, _ = compute_model_flux_all_bands(params, xrt, opt, True, False, None)

FLOOR = 0.05
agg = {}
for d, mf in zip(opt, opt_f):
    g = "7DT (22 x 1pt)" if d['name'].startswith("7DT") else d['name']
    mag_err = (d['flux_err'] / d['flux_mJy']) / LN
    mag_new = np.sqrt(mag_err**2 + FLOOR**2)
    c_now = np.sum(((d['flux_mJy'] - mf) / d['flux_err'])**2)
    e_new = d['flux_mJy'] * mag_new * LN
    c_fl = np.sum(((d['flux_mJy'] - mf) / e_new)**2)
    a = agg.setdefault(g, [0, 0., 0., 0., 0.])
    a[0] += len(d['time']); a[1] += np.sum(1/mag_err**2); a[2] += np.sum(1/mag_new**2)
    a[3] += c_now; a[4] += c_fl
W1 = sum(a[1] for a in agg.values()); W2 = sum(a[2] for a in agg.values())
print(f"{'dataset':16s} {'n':>3} {'share now':>10} {'share w/floor':>14} "
      f"{'chi2/pt now':>12} {'w/floor':>9}")
for g, a in agg.items():
    print(f"{g:16s} {a[0]:3d} {100*a[1]/W1:9.1f}% {100*a[2]/W2:13.1f}% "
          f"{a[3]/a[0]:12.2f} {a[4]/a[0]:9.2f}")
print(f"\ntotal optical chi2: now {sum(a[3] for a in agg.values()):.0f}, "
      f"with 0.05 mag floor {sum(a[4] for a in agg.values()):.0f} "
      f"({sum(a[0] for a in agg.values())} pts)")

# ---- cross-calibration of the new i_data against the AB ensemble ----
print("\n" + "=" * 78)
print("i_data cross-calibration, NEW magnitudes")
print("=" * 78)
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ['upper_limit'].astype(bool)]
lea = circ[(circ.facility == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
trc, mrc = lea.time.to_numpy(float), lea.magnitude.to_numpy(float)
BETA = 0.8
s = (new["time"] >= trc.min()) & (new["time"] <= trc.max())
ref = np.interp(np.log10(new["time"][s]), np.log10(trc), mrc)
off = new["mag_corr"][s] - (ref - 2.5*BETA*np.log10(wl_i/6410.0))
print(f"n={s.sum()} new i_data points inside the Leavitt Rc window")
print(f"  offset vs Leavitt Rc reference = {off.mean():+.3f} mag "
      f"(old value was +0.257)")
print(f"  AB-ensemble anchor was +0.145  ->  i_data sits "
      f"{off.mean()-0.145:+.3f} mag from the AB ensemble (old: -0.112)")
