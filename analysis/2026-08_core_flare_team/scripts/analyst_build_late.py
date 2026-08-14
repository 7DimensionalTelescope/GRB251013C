"""Build the provisional late-time optical dataset.

TWO products:

(A) analyst_late_<band>.csv  -- binned from the pixel-parsed sample.png trails
    (sample_parsed.npz), bands r/i/z, 0.05-dex time bins, as specified.

(B) analyst_late_circular.csv -- the REAL late-time photometry that already
    exists in data/circular.xlsx (t > 3e4 s, 35 measurements out to 1.3e6 s)
    and is currently not used by any fit.  Strictly better than (A) where the
    two overlap.

Both are galactic-extinction-corrected and in mJy.  Points with t > 7e5 s are
written to a *_SNcontaminated.csv sidecar rather than the main file.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.extinction import galactic_extinction
from grb.const import FILTER_INFO
from grb.utils import filter_to_wavelength

OUT = "/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data"
C_AA = 2.99792458e18
ZP_AB = 3631.0
SN_CUT = 7e5
# plotted mag = true apparent mag + offset  (legend of sample.png)
OFFSET = {"r": 0.0, "i": -1.0, "z": -2.0}

npz = np.load(f"{OUT}/sample_parsed.npz")

print("=" * 74)
print("(A) binned from the parsed sample.png trails")
print("=" * 74)
for band in ("r", "i", "z"):
    t = npz[f"{band}_t"]; mplot = npz[f"{band}_m"]
    m_app = mplot - OFFSET[band]                      # undo the legend offset
    wl = filter_to_wavelength(band)                   # FILTER_INFO effective wavelength
    a_gal = float(galactic_extinction(np.array([wl]))[0])
    nu = C_AA / wl

    lt = np.log10(t)
    edges = np.arange(np.floor(lt.min() / 0.05) * 0.05, lt.max() + 0.05, 0.05)
    recs = []
    for lo in edges:
        s = (lt >= lo) & (lt < lo + 0.05)
        if s.sum() < 3:
            continue
        tc = 10 ** (lo + 0.025)
        med = np.median(m_app[s])
        scat = 1.4826 * np.median(np.abs(m_app[s] - med))
        err_mag = max(0.15, scat)
        mcorr = med - a_gal                            # galactic-corrected
        f = ZP_AB * 10 ** (-0.4 * mcorr) * 1e3         # mJy
        fe = f * np.log(10) * 0.4 * err_mag
        recs.append((tc, f, fe, nu, wl, mcorr, err_mag, int(s.sum())))
    df = pd.DataFrame(recs, columns=["time_s", "flux_mJy", "flux_err_mJy",
                                     "frequency_Hz", "wavelength_AA",
                                     "mag_AB_galcorr", "mag_err", "n_pix"])
    keep = df[df.time_s <= SN_CUT]
    drop = df[df.time_s > SN_CUT]
    keep.to_csv(f"{OUT}/analyst_late_{band}.csv", index=False)
    if len(drop):
        drop.to_csv(f"{OUT}/analyst_late_{band}_SNcontaminated.csv", index=False)
    print(f"  {band}: lam={wl:.0f} A  A_gal={a_gal:.3f}  {len(keep)} pts kept "
          f"({keep.time_s.min():.3g}-{keep.time_s.max():.3g} s), {len(drop)} dropped at t>7e5")

print("\n" + "=" * 74)
print("(B) real late-time photometry already in data/circular.xlsx")
print("=" * 74)
raw = pd.read_excel("data/circular.xlsx")
late = raw[(raw["time"] > 3e4) & (~raw["upper_limit"].astype(bool))].copy()
late["wavelength"] = [filter_to_wavelength(str(f)) for f in late["filter"]]
late = late[late["wavelength"] > 0]
# Vega -> AB where FILTER_INFO says the band is Vega (see the zero-point finding)
def ab_correction(filt):
    info = FILTER_INFO.get(str(filt), {})
    zp = info.get("vega_zero_point_jy")
    return 2.5 * np.log10(ZP_AB / zp) if (info.get("system") == "Vega" and zp) else 0.0
late["ab_corr"] = [ab_correction(f) for f in late["filter"]]
late["a_gal"] = [float(galactic_extinction(np.array([w]))[0]) for w in late["wavelength"]]
late["mag_AB_galcorr"] = late["magnitude"] + late["ab_corr"] - late["a_gal"]
late["mag_err"] = np.maximum(late["mag_error"].astype(float), 0.05)
late["flux_mJy"] = ZP_AB * 10 ** (-0.4 * late["mag_AB_galcorr"]) * 1e3
late["flux_err_mJy"] = late["flux_mJy"] * np.log(10) * 0.4 * late["mag_err"]
late["frequency_Hz"] = C_AA / late["wavelength"]
late = late.rename(columns={"time": "time_s"})
cols = ["time_s", "flux_mJy", "flux_err_mJy", "frequency_Hz", "wavelength_AA",
        "mag_AB_galcorr", "mag_err", "facility", "filter", "ab_corr", "Circular"]
late["wavelength_AA"] = late["wavelength"]
keep = late[late.time_s <= SN_CUT][cols].sort_values("time_s")
drop = late[late.time_s > SN_CUT][cols].sort_values("time_s")
keep.to_csv(f"{OUT}/analyst_late_circular.csv", index=False)
drop.to_csv(f"{OUT}/analyst_late_circular_SNcontaminated.csv", index=False)
print(keep.to_string(index=False))
print(f"\n{len(keep)} kept, {len(drop)} moved to the SN-contaminated sidecar (t>7e5 s)")
print("\nSN-contaminated (t>7e5 s):")
print(drop.to_string(index=False))
