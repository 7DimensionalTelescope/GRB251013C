"""Decisive test of the Vega/AB zero-point question.

grb/utils._mag_to_flux_mJy converts EVERY band with zero_point=3631 Jy (AB).
const.FILTER_INFO declares Rc/Ic/R/V/J as Vega with their own zero points.
If the reported magnitudes really are Vega, then converting them as AB makes
their fluxes too bright by  dm = 2.5*log10(3631/ZP_vega)  mag, and that error
must show up as a per-FILTER (not per-facility) offset whose size tracks dm.

V is the control: ZP_vega(V)=3640 ~ 3631, so dm(V) ~ 0 and V must show no
offset even though it is a Vega band.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.io import read_data
from grb.const import FILTER_INFO

ZP_AB = 3631.0
WL_RC = 6410.0

circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ["upper_limit"].astype(bool)]
lea_rc = circ[(circ.facility == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
trc, mrc = lea_rc["time"].to_numpy(float), lea_rc["magnitude"].to_numpy(float)
TMIN, TMAX = trc.min(), trc.max()

idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)

rows = []
for _, r in circ.iterrows():
    wl = float(r["wavelength"])
    if wl > 0 and TMIN <= r["time"] <= TMAX:
        rows.append((r["facility"], str(r["filter"]), float(r["time"]),
                     float(r["magnitude"]), wl))
for _, r in idat.iterrows():
    if TMIN <= r["time"] <= TMAX:
        rows.append(("i_data", "i_data", float(r["time"]), float(r["magnitude"]),
                     float(r["wavelength"])))
df = pd.DataFrame(rows, columns=["facility", "filter", "time", "mag", "wl"])
df = df[df["filter"] != "L"]        # GOTO 'L': FILTER_INFO maps it to 34500 A (IR), clearly wrong
df = df[df["filter"] != "R"]        # single VTP point with no quoted error

print("Reference light curve = Leavitt Rc (35 pts). Offsets are relative to it,")
print("after a colour correction with F_nu ~ nu^-beta.\n")

for BETA in (0.8, 1.0, 1.2):
    ref = np.interp(np.log10(df["time"]), np.log10(trc), mrc)
    df["off"] = df["mag"] - (ref - 2.5 * BETA * np.log10(df["wl"] / WL_RC))

    # AB-system ensemble = the anchor everything is measured against
    ab_filters = ["g", "g'", "r", "r'", "i", "z"]
    ab = df[df["filter"].isin(ab_filters)]
    anchor = ab["off"].mean()

    print("=" * 74)
    print(f"beta = {BETA}:  AB-system ensemble anchor = {anchor:+.3f} "
          f"(n={len(ab)}, scatter {ab['off'].std(ddof=1):.3f})")
    print("=" * 74)
    print(f"{'filter':8s} {'system':6s} {'ZP_Jy':>7} {'n':>3} {'dm=2.5log(3631/ZP)':>19} "
          f"{'measured offset':>16} {'residual':>9}")
    for filt in ("g", "r", "i", "z", "i_data", "V", "Rc", "Ic"):
        sub = df[df["filter"] == filt]
        if len(sub) == 0:
            continue
        key = "i" if filt == "i_data" else filt
        info = FILTER_INFO.get(key, {})
        zp = info.get("vega_zero_point_jy")
        sysname = info.get("system", "?")
        dm = 2.5 * np.log10(ZP_AB / zp) if zp else 0.0
        meas = anchor - sub["off"].mean()     # how much brighter than the AB ensemble
        print(f"{filt:8s} {sysname:6s} {str(zp) if zp else '3631':>7} {len(sub):3d} "
              f"{dm:19.3f} {meas:16.3f} {meas-dm:9.3f}")
    print()

print("Reading: 'measured offset' is how much brighter than the AB ensemble that")
print("filter's points sit when converted with the AB zero point.  If the band is")
print("really Vega, that should equal dm.  'residual' is the leftover mismatch.")
