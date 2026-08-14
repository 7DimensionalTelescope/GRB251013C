"""How did i_data actually change?  Match old<->new by time, not by row index."""
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

new = pd.read_csv(f"{MAIN}/data/i_data.csv")
trigger_mjd = Time(TRIGGER_TIME.strftime("%Y-%m-%dT%H:%M:%S")).mjd
new["time"] = (new["mjd"].to_numpy(float) - trigger_mjd) * 86400.0
new = new.rename(columns={"mag": "magnitude", "magerr": "mag_error"}).sort_values("time")
a_gal = float(galactic_extinction(np.array([filter_to_wavelength("i")]))[0])
new["mag_corr"] = new["magnitude"] - a_gal
old = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)

to, mo = old["time"].to_numpy(float), old["magnitude"].to_numpy(float)
tn, mn = new["time"].to_numpy(float), new["mag_corr"].to_numpy(float)

print("nearest-time matching old -> new")
print(f"{'t_old':>9} {'t_new':>9} {'dt':>7} {'m_old':>7} {'m_new':>7} {'dm':>7}")
dms, dts = [], []
for i in range(len(to)):
    j = np.argmin(np.abs(tn - to[i]))
    dm = mn[j] - mo[i]; dt = tn[j] - to[i]
    dms.append(dm); dts.append(dt)
    if abs(dm) > 0.08 or abs(dt) > 30:
        print(f"{to[i]:9.1f} {tn[j]:9.1f} {dt:+7.1f} {mo[i]:7.3f} {mn[j]:7.3f} {dm:+7.3f}")
dms, dts = np.array(dms), np.array(dts)
print(f"\nafter time-matching: dm mean {dms.mean():+.4f}, rms {dms.std():.4f}, "
      f"max|dm| {np.abs(dms).max():.3f}")
print(f"                     dt mean {dts.mean():+.2f} s, max|dt| {np.abs(dts).max():.1f} s")

print("\ntime-dependence of the recalibration (new - old, mag):")
for lo, hi in ((90, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 11000)):
    s = (to >= lo) & (to < hi)
    if s.sum():
        print(f"  {lo:6d}-{hi:6d} s: n={s.sum():2d}  mean dm {dms[s].mean():+.3f}  "
              f"rms {dms[s].std():.3f}")

print("\n" + "=" * 74)
print("does the new i_data duplicate rows already in circular.xlsx?")
print("=" * 74)
circ = pd.read_excel("data/circular.xlsx")
nut = circ[circ["facility"].astype(str).str.strip() == "NUTTelA-TAO"]
print(f"circular.xlsx NUTTelA-TAO rows: {len(nut)}, filters {sorted(nut['filter'].unique())}, "
      f"t {nut['time'].min():.0f}-{nut['time'].max():.0f} s")
nut_i = nut[nut["filter"] == "i"]
print(f"  of which filter 'i': {len(nut_i)} rows at t = "
      f"{np.round(nut_i['time'].to_numpy(float), 0)}")
match = 0
for t in nut_i["time"].to_numpy(float):
    if np.min(np.abs(tn - t)) < 60:
        match += 1
print(f"  {match} of {len(nut_i)} coincide with an i_data epoch within 60 s")
print("  -> i_data IS the NUTTelA-TAO i series; only Leavitt is taken from")
print("     circular.xlsx today, so there is no double-count, but any future")
print("     wholesale ingest of circular.xlsx would duplicate these points.")
