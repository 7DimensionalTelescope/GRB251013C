"""Re-run the photometric-system test with the NEW i_data included, and do the
direct same-band i-vs-i comparisons."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
MAIN = "/data/dtak/research/grb/GRB251013C"
sys.path.insert(0, WT); os.chdir(WT)
from astropy.time import Time
from grb.const import TRIGGER_TIME, FILTER_INFO
from grb.extinction import galactic_extinction
from grb.utils import filter_to_wavelength
from grb.io import read_data

ZP_AB, WL_RC, BETA = 3631.0, 6410.0, 0.8
new = pd.read_csv(f"{MAIN}/data/i_data.csv")
trigger_mjd = Time(TRIGGER_TIME.strftime("%Y-%m-%dT%H:%M:%S")).mjd
new["time"] = (new["mjd"].to_numpy(float) - trigger_mjd) * 86400.0
new = new.rename(columns={"mag": "magnitude"}).sort_values("time")
WL_I = filter_to_wavelength("i")
new["mag_corr"] = new["magnitude"] - float(galactic_extinction(np.array([WL_I]))[0])

circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ["upper_limit"].astype(bool)]
lea = circ[(circ.facility == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
trc, mrc = lea.time.to_numpy(float), lea.magnitude.to_numpy(float)
TMIN, TMAX = trc.min(), trc.max()

rows = []
for _, r in circ.iterrows():
    if float(r["wavelength"]) > 0 and TMIN <= r["time"] <= TMAX and str(r["filter"]) not in ("L", "R"):
        rows.append((r["facility"], str(r["filter"]), float(r["time"]),
                     float(r["magnitude"]), float(r["wavelength"])))
for _, r in new.iterrows():
    if TMIN <= r["time"] <= TMAX:
        rows.append(("NUTTelA-TAO", "i_data", float(r["time"]), float(r["mag_corr"]), WL_I))
df = pd.DataFrame(rows, columns=["facility", "filter", "time", "mag", "wl"])
ref = np.interp(np.log10(df["time"]), np.log10(trc), mrc)
df["off"] = df["mag"] - (ref - 2.5 * BETA * np.log10(df["wl"] / WL_RC))
fac = df.groupby(["filter", "facility"])["off"].mean().reset_index()
ab = ["g", "g'", "r", "r'", "i", "z"]
anchor = fac[fac["filter"].isin(ab)]["off"].median()

print("=" * 76)
print(f"photometric-system test, NEW i_data (AB anchor = {anchor:+.3f}, "
      f"{len(fac[fac['filter'].isin(ab)])} facility/filter combos)")
print("=" * 76)
print(f"{'filter':9s} {'system':6s} {'nfac':>4} {'predicted':>10} {'measured':>9} {'residual':>9}")
for filt in ("g", "r", "i", "z", "i_data", "V", "Rc", "Ic"):
    sub = fac[fac["filter"] == filt]
    if len(sub) == 0:
        continue
    info = FILTER_INFO.get("i" if filt == "i_data" else filt, {})
    zp = info.get("vega_zero_point_jy")
    dm = 2.5 * np.log10(ZP_AB / zp) if (info.get("system") == "Vega" and zp) else 0.0
    meas = anchor - sub["off"].median()
    print(f"{filt:9s} {info.get('system','?'):6s} {len(sub):4d} {dm:10.3f} "
          f"{meas:9.3f} {meas-dm:9.3f}")

print("\n" + "=" * 76)
print("direct same-band comparisons: other facilities' i vs the NEW i_data")
print("=" * 76)
ti, mi = new["time"].to_numpy(float), new["mag_corr"].to_numpy(float)
oi = circ[(circ["filter"] == "i") & (circ.facility != "NUTTelA-TAO")]
for _, r in oi.iterrows():
    if ti.min() <= r["time"] <= ti.max():
        m = np.interp(np.log10(r["time"]), np.log10(ti), mi)
        print(f"  t={r['time']:8.0f}  {r['facility'][:22]:22s} i={r['magnitude']:.3f}  "
              f"i_data(interp)={m:.3f}  diff={r['magnitude']-m:+.3f}")
print("\n  (positive diff = the other facility is fainter than NUTTelA-TAO)")
