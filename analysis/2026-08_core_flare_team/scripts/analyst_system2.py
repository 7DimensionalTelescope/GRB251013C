"""Same test as analyst_system.py but robust: aggregate per facility first,
then take the median across facilities, so one badly-calibrated observatory
cannot drive a filter's answer."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.io import read_data
from grb.const import FILTER_INFO

ZP_AB, WL_RC, BETA = 3631.0, 6410.0, 0.8
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ["upper_limit"].astype(bool)]
lea_rc = circ[(circ.facility == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
trc, mrc = lea_rc["time"].to_numpy(float), lea_rc["magnitude"].to_numpy(float)
TMIN, TMAX = trc.min(), trc.max()
idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)

rows = []
for _, r in circ.iterrows():
    if float(r["wavelength"]) > 0 and TMIN <= r["time"] <= TMAX and str(r["filter"]) != "L":
        rows.append((r["facility"], str(r["filter"]), float(r["time"]),
                     float(r["magnitude"]), float(r["wavelength"])))
for _, r in idat.iterrows():
    if TMIN <= r["time"] <= TMAX:
        rows.append(("i_data", "i_data", float(r["time"]), float(r["magnitude"]),
                     float(r["wavelength"])))
df = pd.DataFrame(rows, columns=["facility", "filter", "time", "mag", "wl"])
ref = np.interp(np.log10(df["time"]), np.log10(trc), mrc)
df["off"] = df["mag"] - (ref - 2.5 * BETA * np.log10(df["wl"] / WL_RC))

# per facility+filter means, then median across facilities within a filter
fac = df.groupby(["filter", "facility"])["off"].mean().reset_index()
print("per facility+filter mean offset relative to Leavitt Rc (beta=0.8):")
for _, r in fac.sort_values(["filter", "facility"]).iterrows():
    print(f"  {r['filter']:8s} {r['facility'][:38]:38s} {r['off']:+.3f}")

ab_filters = ["g", "g'", "r", "r'", "i", "z"]
anchor = fac[fac["filter"].isin(ab_filters)]["off"].median()
print(f"\nAB-system anchor (median over {len(fac[fac['filter'].isin(ab_filters)])} "
      f"facility/filter combos) = {anchor:+.3f}")

print(f"\n{'filter':8s} {'system':6s} {'nfac':>4} {'predicted dm':>13} {'measured':>9} {'residual':>9}")
for filt in ("g", "r", "i", "z", "i_data", "V", "Rc", "Ic"):
    sub = fac[fac["filter"] == filt]
    if len(sub) == 0:
        continue
    info = FILTER_INFO.get("i" if filt == "i_data" else filt, {})
    zp = info.get("vega_zero_point_jy")
    dm = 2.5 * np.log10(ZP_AB / zp) if zp else 0.0
    meas = anchor - sub["off"].median()
    print(f"{filt:8s} {info.get('system','?'):6s} {len(sub):4d} {dm:13.3f} "
          f"{meas:9.3f} {meas-dm:9.3f}")
